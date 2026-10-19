import os
import sys
import time
import json
from dotenv import load_dotenv
from openai import OpenAI
import psutil
import pandas as pd

load_dotenv()

API_KEY = os.getenv("LLM_API_KEY")
BASE_URL = os.getenv("LLM_BASE_URL")

def load_models_from_env():
    """Загружает список моделей из .env (LLM_MODEL_1, LLM_MODEL_2 и т.д.)"""
    models = []
    i = 1
    while True:
        name = os.getenv(f"LLM_MODEL_{i}")
        if not name:
            break
        models.append(name)
        i += 1
    return models

class LocalLLMTester:
    def __init__(self, model_name):
        self.client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        self.model = model_name

    def ask_stream(self, prompt, temperature=0.2):
        """
        Запрос с использованием стриминга для замера TTFT (Time To First Token).
        """
        messages = [{"role": "user", "content": prompt}]
        
        t0 = time.perf_counter()
        ttft = None
        full_text = ""
        out_tokens = 0
        in_tokens = 0
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=True, 
                stream_options={"include_usage": True} 
            )
            
            for chunk in response:
                if ttft is None and chunk.choices and chunk.choices[0].delta.content:
                    ttft = time.perf_counter() - t0
                
                if chunk.choices and chunk.choices[0].delta.content:
                    full_text += chunk.choices[0].delta.content
                
                if hasattr(chunk, 'usage') and chunk.usage:
                    in_tokens = chunk.usage.prompt_tokens
                    out_tokens = chunk.usage.completion_tokens

            total_time = time.perf_counter() - t0
            
            gen_time = total_time - (ttft or 0)
            speed = round(out_tokens / gen_time, 2) if gen_time > 0 else 0
            
            return {
                "text": full_text,
                "ttft": round(ttft, 3) if ttft else None,
                "total_time": round(total_time, 3),
                "in_tokens": in_tokens,
                "out_tokens": out_tokens,
                "tokens_per_sec": speed,
                "error": None
            }
        except Exception as e:
            return {"text": None, "ttft": None, "total_time": None, "error": str(e)}

    def run_tests(self):
        print(f"\n{'='*60}")
        print(f"ТЕСТИРОВАНИЕ МОДЕЛИ: {self.model}")
        print(f"{'='*60}")
        
        prompts = [
            "Объясни в 5 предложениях разницу между хэшированием и шифрованием.",
            "Суммаризируй следующий текст в 3 пунктах: Информационная безопасность — это практика защиты информации от несанкционированного доступа, использования, раскрытия, изменения или уничтожения. Она включает в себя защиту как аппаратного, так и программного обеспечения.",
            "Определи категорию события: «пять неуспешных входов, затем успешный вход с нового IP». Ответ: норма / подозрительно / инцидент и краткое объяснение.",
            'Верни JSON с полями risk_level, reason, recommended_action для указанного события: "пять неуспешных входов, затем успешный вход с нового IP".',
            "Напиши функцию Python, проверяющую, содержит ли строка IPv4-адрес. Только код."
        ]

        results = []
        for i, prompt in enumerate(prompts, start=1):
            print(f"  Запрос {i}/5...", end=" ", flush=True)
            
            # Замер RAM до запроса
            ram_before = psutil.virtual_memory().used / (1024**3)
            
            res = self.ask_stream(prompt)
            
            # Замер RAM после запроса
            ram_after = psutil.virtual_memory().used / (1024**3)
            ram_delta = round(ram_after - ram_before, 2)
            
            if res["error"]:
                print(f"ОШИБКА: {res['error'][:50]}")
            else:
                print(f"OK (TTFT: {res['ttft']}с, Скорость: {res['tokens_per_sec']} т/с, Токенов: {res['out_tokens']})")
                
            results.append({
                "model": self.model,
                "prompt_id": i,
                "latency_total": res.get("total_time"),
                "ttft": res.get("ttft"),
                "tokens_per_sec": res.get("tokens_per_sec"),
                "out_tokens": res.get("out_tokens"),
                "ram_delta_gb": ram_delta,
                "answer": res.get("text"),
                "error": res.get("error")
            })
            
        return results

if __name__ == "__main__":
    models = load_models_from_env()
    
    if not models:
        print("Ошибка: В файле .env не найдены модели (LLM_MODEL_1 и т.д.)")
        sys.exit(1)

    print("\nДоступные модели для теста:")
    for idx, m in enumerate(models, start=1):
        print(f"  [{idx}] {m}")
        
    # Выбор модели (через аргумент или интерактивно)
    if len(sys.argv) > 1:
        choice = sys.argv[1].strip()
    else:
        choice = input("\nВыберите номер модели для тестирования: ").strip()

    try:
        model_idx = int(choice) - 1
        if not (0 <= model_idx < len(models)):
            raise ValueError
    except ValueError:
        print("Ошибка: Введите корректный номер из списка.")
        sys.exit(1)

    selected_model = models[model_idx]
    
    # Запуск
    tester = LocalLLMTester(selected_model)
    results = tester.run_tests()
    
    # Сохранение
    clean_name = selected_model.replace('/', '_').replace('\\', '_')
    filename = f"results_{clean_name}.csv"
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"\nРезультаты успешно сохранены в файл: {filename}")