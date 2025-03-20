import os
import subprocess

# Папки для анализа
directories = ["core", "."]

# Файл для сохранения результатов
report_file = "analysis_report.txt"

# Команда для выполнения анализа
command = ["flake8", "--max-line-length=120"] + directories

# Выполнение команды и сохранение результатов в файл
with open(report_file, "w") as file:
    result = subprocess.run(command, stdout=file, stderr=subprocess.STDOUT)

print(f"Результаты анализа сохранены в {report_file}")
