import os
import subprocess

# ����� ��� �������
directories = ["core", "."]

# ���� ��� ���������� �����������
report_file = "analysis_report.txt"

# ������� ��� ���������� �������
command = ["flake8", "--max-line-length=120"] + directories

# ���������� ������� � ���������� ����������� � ����
with open(report_file, "w") as file:
    result = subprocess.run(command, stdout=file, stderr=subprocess.STDOUT)

print(f"���������� ������� ��������� � {report_file}")
