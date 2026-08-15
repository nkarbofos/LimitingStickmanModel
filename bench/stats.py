"""Статистика задержек и проверка наличия моделей."""

import os
import statistics


def print_latency_stats(title, latencies):
    if not latencies:
        print(f"{title}: нет данных")
        return
    mean_ms = statistics.mean(latencies)
    median_ms = statistics.median(latencies)
    std_ms = statistics.stdev(latencies) if len(latencies) > 1 else 0.0
    p95_ms = sorted(latencies)[int(0.95 * (len(latencies) - 1))]
    print(f"\n{title}:")
    print(f"  mean:   {mean_ms:>8.2f} ms  ->  {1000/mean_ms:>6.1f} FPS")
    print(f"  median: {median_ms:>8.2f} ms  ->  {1000/median_ms:>6.1f} FPS")
    print(f"  std:    {std_ms:>8.2f} ms")
    print(f"  min:    {min(latencies):>8.2f} ms")
    print(f"  max:    {max(latencies):>8.2f} ms")
    print(f"  p95:    {p95_ms:>8.2f} ms")


def check_model(path, hint):
    if not os.path.exists(path):
        print(f"[!] Модель не найдена: {path}\n    {hint}")
        return False
    return True
