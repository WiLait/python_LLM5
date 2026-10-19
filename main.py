import os
import time
import json
from dotenv import load_dotenv
from openai import OpenAI
import psutil
import pandas as pd

load_dotenv()

API_KEY = os.getenv("LLM_API_KEY")
BASE_URL = os.getenv("LLM_BASE_URL")

class LocalLLMTester:
    def __init__(self, model_name):
        self.client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        self.model = model_name

    def ask_stream(self, prompt, temperature=0.2):
        """
        Запрос с использованием стриминга для замера TTFT (Time To First Token).
        Возвращает полный текст, TTFT, общее время, токены.
        """
        messages = [{"role": "user", "content": prompt}]
        
        # Замер начального времени
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
                stream=True, # Включаем стриминг
                stream_options={"include_usage": True} # Просим LM Studio вернуть токены в конце
            )
            
            for chunk in response:
                if ttft is None and chunk.choices and chunk.choices[0].delta.content:
                    # Фиксируем время первого токена
                    ttft = time.perf_counter() - t0
                
                if chunk.choices and chunk.choices[0].delta.content:
                    full_text += chunk.choices[0].delta.content
                
                if hasattr(chunk, 'usage') and chunk.usage:
                    in_tokens = chunk.usage.prompt_tokens
                    out_tokens = chunk.usage.completion_tokens

            total_time = time.perf_counter() - t0
            
            # Скорость генерации = токены / время после первого токена
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
        print(f"\n=== Тестируем модель: {self.model} ===")
        
        # 5 запросов из методички ЛР5 (Задание 2)
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
            
            # Замер RAM до
            ram_before = psutil.virtual_memory().used / (1024**3)
            
            res = self.ask_stream(prompt)
            
            # Замер RAM после
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
    model_name = input("Введите точное имя модели (как в LM Studio): ").strip()
    
    tester = LocalLLMTester(model_name)
    results = tester.run_tests()
    
    df = pd.DataFrame(results)
    filename = f"results_{model_name.replace('/', '_')}.csv"
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"\nРезультаты сохранены в {filename}")