"""
Модуль для запуска pytest и парсинга результатов.

Предоставляет интерфейс для:
    - Запуска тестов через pytest
    - Получения coverage отчётов
    - Парсинга результатов тестов
    - Real-time callbacks для обновления UI
"""

import subprocess
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass


@dataclass
class TestResult:
    """Результат выполнения одного теста."""
    name: str
    status: str  # 'passed', 'failed', 'skipped'
    duration_sec: float
    error: Optional[str] = None
    location: Optional[str] = None


class TestRunner:
    """
    Обёртка для запуска pytest и анализа результатов.

    Примеры использования:
        >>> runner = TestRunner(tests_dir="tests/")
        >>>
        >>> # Запуск всех тестов с coverage
        >>> results = runner.run_tests(test_pattern="test_*.py", with_coverage=True)
        >>> print(f"Passed: {results['passed']}/{results['total_tests']}")
        >>>
        >>> # Получение coverage отчёта
        >>> coverage = runner.get_coverage_report()
        >>> for module, pct in coverage.items():
        >>>     print(f"{module}: {pct:.1f}%")
    """

    def __init__(self, tests_dir: str = "tests/"):
        """
        Инициализация test runner.

        Args:
            tests_dir: Директория с тестами
        """
        self.tests_dir = Path(tests_dir)
        self.last_results = None
        self.last_coverage = None

    def run_tests(
        self,
        test_pattern: str = "test_*.py",
        with_coverage: bool = True,
        callback: Optional[Callable[[int, int], None]] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Запускает тесты и возвращает результаты.

        Args:
            test_pattern: Паттерн для выбора тестов (например, "test_transformer.py")
            with_coverage: Запускать с coverage анализом
            callback: Функция для real-time обновления (current, total)
            verbose: Детальный вывод

        Returns:
            {
                'total_tests': int,
                'passed': int,
                'failed': int,
                'skipped': int,
                'duration_sec': float,
                'coverage_pct': float,  # Если with_coverage=True
                'failed_tests': List[TestResult],
                'all_tests': List[TestResult]
            }
        """
        # Формируем команду pytest
        cmd = ['pytest']

        # Добавляем путь к тестам
        if test_pattern and test_pattern != "test_*.py":
            test_path = self.tests_dir / test_pattern
            cmd.append(str(test_path))
        else:
            cmd.append(str(self.tests_dir))

        # Опции pytest
        if verbose:
            cmd.append('-v')

        cmd.extend(['--tb=short', '--color=yes'])

        # Coverage
        if with_coverage:
            cmd.extend(['--cov=src/python', '--cov-report=json'])

        # JSON output для парсинга (если доступен pytest-json-report)
        # cmd.append('--json-report')

        # Запускаем pytest
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 минут максимум
            )

            # Парсим вывод pytest
            parsed_results = self._parse_pytest_output(result.stdout, result.stderr)

            # Парсим coverage (если включён)
            coverage_pct = 0.0
            if with_coverage:
                coverage_pct = self._parse_coverage()

            parsed_results['coverage_pct'] = coverage_pct

            self.last_results = parsed_results
            return parsed_results

        except subprocess.TimeoutExpired:
            return {
                'total_tests': 0,
                'passed': 0,
                'failed': 0,
                'skipped': 0,
                'duration_sec': 0.0,
                'coverage_pct': 0.0,
                'failed_tests': [],
                'all_tests': [],
                'error': 'Test execution timed out (>300 sec)'
            }

        except Exception as e:
            return {
                'total_tests': 0,
                'passed': 0,
                'failed': 0,
                'skipped': 0,
                'duration_sec': 0.0,
                'coverage_pct': 0.0,
                'failed_tests': [],
                'all_tests': [],
                'error': str(e)
            }

    def _parse_pytest_output(self, stdout: str, stderr: str) -> Dict[str, Any]:
        """
        Парсит вывод pytest для извлечения результатов.

        Args:
            stdout: Стандартный вывод pytest
            stderr: Вывод ошибок pytest

        Returns:
            Словарь с результатами тестов
        """
        # Regex для парсинга строки вида: "211 passed in 2.67s"
        summary_pattern = r'(\d+) passed(?:, (\d+) failed)?(?:, (\d+) skipped)? in ([\d.]+)s'
        match = re.search(summary_pattern, stdout)

        if match:
            passed = int(match.group(1))
            failed = int(match.group(2)) if match.group(2) else 0
            skipped = int(match.group(3)) if match.group(3) else 0
            duration = float(match.group(4))
        else:
            # Fallback: если не нашли summary
            passed = stdout.count(' PASSED')
            failed = stdout.count(' FAILED')
            skipped = stdout.count(' SKIPPED')
            duration = 0.0

        total = passed + failed + skipped

        # Парсим провалившиеся тесты
        failed_tests = self._parse_failed_tests(stdout)

        return {
            'total_tests': total,
            'passed': passed,
            'failed': failed,
            'skipped': skipped,
            'duration_sec': duration,
            'failed_tests': failed_tests,
            'all_tests': []  # Можно добавить парсинг всех тестов при необходимости
        }

    def _parse_failed_tests(self, output: str) -> List[Dict[str, str]]:
        """
        Парсит информацию о провалившихся тестах.

        Args:
            output: Вывод pytest

        Returns:
            Список словарей с информацией о failed тестах
        """
        failed_tests = []

        # Regex для парсинга failed тестов
        # Формат: tests/test_file.py::test_name FAILED
        failed_pattern = r'(tests/[\w/]+\.py)::(test_\w+)\s+FAILED'

        matches = re.finditer(failed_pattern, output)

        for match in matches:
            location = match.group(1)
            test_name = match.group(2)

            # Ищем сообщение об ошибке (упрощённый поиск)
            error_msg = self._extract_error_message(output, test_name)

            failed_tests.append({
                'name': test_name,
                'location': location,
                'error': error_msg
            })

        return failed_tests

    def _extract_error_message(self, output: str, test_name: str) -> str:
        """Извлекает сообщение об ошибке для конкретного теста."""
        # Ищем секцию FAILURES
        failures_section_pattern = r'_{5,} (FAILURES|ERRORS) _{5,}(.*?)(?=_{5,}|$)'
        failures_match = re.search(failures_section_pattern, output, re.DOTALL)

        if failures_match:
            failures_text = failures_match.group(2)

            # Ищем конкретный тест
            test_section_pattern = rf'_{5,} {test_name} _{5,}(.*?)(?=_{5,}|\Z)'
            test_match = re.search(test_section_pattern, failures_text, re.DOTALL)

            if test_match:
                error_text = test_match.group(1).strip()
                # Берём первые 200 символов
                return error_text[:200] if len(error_text) > 200 else error_text

        return "No error details available"

    def _parse_coverage(self) -> float:
        """
        Парсит coverage отчёт из coverage.json.

        Returns:
            Процент coverage (0.0-100.0)
        """
        coverage_file = Path('coverage.json')

        if not coverage_file.exists():
            return 0.0

        try:
            with open(coverage_file, 'r') as f:
                coverage_data = json.load(f)

            # Coverage report формат: {'totals': {'percent_covered': 85.5}}
            if 'totals' in coverage_data:
                return coverage_data['totals'].get('percent_covered', 0.0)

            return 0.0

        except Exception as e:
            print(f"Error parsing coverage: {e}")
            return 0.0

    def get_coverage_report(self) -> Dict[str, float]:
        """
        Возвращает детальный coverage report по модулям.

        Returns:
            {
                'src/python/models/expert.py': 95.2,
                'src/python/routing/router.py': 87.3,
                ...
            }
        """
        coverage_file = Path('coverage.json')

        if not coverage_file.exists():
            return {}

        try:
            with open(coverage_file, 'r') as f:
                coverage_data = json.load(f)

            # Извлекаем покрытие по файлам
            files_coverage = {}

            if 'files' in coverage_data:
                for filepath, file_data in coverage_data['files'].items():
                    # Получаем процент покрытия
                    summary = file_data.get('summary', {})
                    percent = summary.get('percent_covered', 0.0)

                    # Убираем абсолютный путь, оставляем относительный
                    rel_path = filepath
                    if filepath.startswith('/'):
                        # Попытка извлечь относительный путь
                        parts = Path(filepath).parts
                        if 'src' in parts:
                            idx = parts.index('src')
                            rel_path = str(Path(*parts[idx:]))

                    files_coverage[rel_path] = percent

            self.last_coverage = files_coverage
            return files_coverage

        except Exception as e:
            print(f"Error parsing coverage report: {e}")
            return {}

    def get_test_list(self) -> List[Dict[str, str]]:
        """
        Возвращает список всех доступных тестов.

        Returns:
            [
                {'path': 'test_transformer.py', 'name': 'Transformer tests'},
                {'path': 'test_expert.py', 'name': 'ExpertModel tests'},
                ...
            ]
        """
        test_files = []

        if not self.tests_dir.exists():
            return test_files

        # Ищем все test_*.py файлы
        for test_file in sorted(self.tests_dir.glob('test_*.py')):
            # Извлекаем название модуля из имени файла
            module_name = test_file.stem.replace('test_', '').replace('_', ' ').title()

            test_files.append({
                'path': test_file.name,
                'name': f"{module_name} tests",
                'full_path': str(test_file)
            })

        # Добавляем опцию "Все тесты"
        test_files.insert(0, {
            'path': 'test_*.py',
            'name': 'Все тесты',
            'full_path': str(self.tests_dir)
        })

        return test_files

    def get_test_count(self, test_pattern: str = "test_*.py") -> int:
        """
        Возвращает количество тестов для данного паттерна.

        Args:
            test_pattern: Паттерн файла тестов

        Returns:
            Количество тестов
        """
        # Запускаем pytest с --collect-only для подсчёта тестов
        cmd = ['pytest', '--collect-only', '-q']

        if test_pattern != "test_*.py":
            test_path = self.tests_dir / test_pattern
            cmd.append(str(test_path))
        else:
            cmd.append(str(self.tests_dir))

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10
            )

            # Парсим вывод: "211 tests collected in 0.05s"
            match = re.search(r'(\d+) tests? collected', result.stdout)
            if match:
                return int(match.group(1))

            return 0

        except Exception:
            return 0
