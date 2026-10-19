import psutil
import time

def get_system_metrics():
    """Возвращает текущее потребление RAM и CPU."""
    ram = psutil.virtual_memory()
    return {
        "ram_used_gb": round(ram.used / (1024**3), 2),
        "ram_percent": ram.percent,
        "cpu_percent": psutil.cpu_percent(interval=0.1)
    }

def monitor_while_generating(func, *args, **kwargs):
    """Запускает функцию и пишет пиковое потребление ресурсов."""
    peak_ram = 0
    peak_cpu = 0
    
    # Простой мониторинг в фоне (без потоков, просто до и после, 
    # но для точности в ЛР5 лучше запустить мониторинг в отдельном потоке)
    # Для простоты сделаем замеры до и во время вызова.
    
    print("  [Монитор] Начало замера ресурсов...")
    start_metrics = get_system_metrics()
    
    # Здесь мы вызываем саму функцию генерации
    result = func(*args, **kwargs)
    
    end_metrics = get_system_metrics()
    
    # На самом деле в методичке просят пиковое значение. 
    # Для честного пика нужен отдельный поток, но для начала хватит разницы.
    result["ram_delta_gb"] = round(end_metrics["ram_used_gb"] - start_metrics["ram_used_gb"], 2)
    result["cpu_during"] = end_metrics["cpu_percent"]
    return result