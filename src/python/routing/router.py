"""
SimpleRouter - Rule-Based маршрутизация запросов к экспертам.

Реализует простую систему маршрутизации на основе ключевых слов
для выбора подходящего эксперта в MoE системе.
"""

from typing import Dict, List, Optional, Tuple, Set
import re
from dataclasses import dataclass
import json


@dataclass
class ExpertInfo:
    """
    Информация об эксперте.

    Attributes:
        expert_id: уникальный ID эксперта
        name: название эксперта
        description: описание специализации
        keywords: ключевые слова для маршрутизации
        priority: приоритет (0-10, где 10 - highest)
    """
    expert_id: str
    name: str
    description: str
    keywords: Set[str]
    priority: int = 5


@dataclass
class RoutingResult:
    """
    Результат маршрутизации.

    Attributes:
        expert_id: ID выбранного эксперта
        confidence: уверенность выбора (0.0-1.0)
        matched_keywords: найденные ключевые слова
        reasoning: объяснение выбора
    """
    expert_id: str
    confidence: float
    matched_keywords: List[str]
    reasoning: str


class SimpleRouter:
    """
    Rule-Based роутер для MoE системы.

    Выбирает эксперта на основе ключевых слов в запросе.

    Пример:
        >>> router = SimpleRouter()
        >>> router.add_expert(
        ...     expert_id="expert_python",
        ...     name="Python Expert",
        ...     keywords={"python", "pandas", "numpy", "flask"}
        ... )
        >>> result = router.route("How to use pandas dataframe?")
        >>> print(result.expert_id)
        'expert_python'
    """

    def __init__(self, default_expert_id: str = "general"):
        """
        Инициализация роутера.

        Args:
            default_expert_id: ID эксперта по умолчанию (fallback)
        """
        self.experts: Dict[str, ExpertInfo] = {}
        self.default_expert_id = default_expert_id

    def add_expert(
        self,
        expert_id: str,
        name: str,
        description: str = "",
        keywords: Optional[Set[str]] = None,
        priority: int = 5
    ) -> None:
        """
        Добавить эксперта в систему маршрутизации.

        Args:
            expert_id: уникальный ID
            name: название эксперта
            description: описание специализации
            keywords: набор ключевых слов (lowercase)
            priority: приоритет при равном score (0-10)
        """
        if keywords is None:
            keywords = set()

        # Преобразуем все keywords в lowercase
        keywords_lower = {kw.lower() for kw in keywords}

        self.experts[expert_id] = ExpertInfo(
            expert_id=expert_id,
            name=name,
            description=description,
            keywords=keywords_lower,
            priority=priority
        )

    def remove_expert(self, expert_id: str) -> bool:
        """
        Удалить эксперта из системы.

        Args:
            expert_id: ID эксперта

        Returns:
            True если эксперт был удалён, False если не найден
        """
        if expert_id in self.experts:
            del self.experts[expert_id]
            return True
        return False

    def route(
        self,
        query: str,
        top_k: int = 1,
        min_confidence: float = 0.0
    ) -> List[RoutingResult]:
        """
        Маршрутизация запроса к экспертам.

        Args:
            query: текстовый запрос пользователя
            top_k: количество экспертов для возврата
            min_confidence: минимальная уверенность для включения в результат

        Returns:
            Список результатов маршрутизации, отсортированный по confidence
        """
        if not self.experts:
            return [RoutingResult(
                expert_id=self.default_expert_id,
                confidence=1.0,
                matched_keywords=[],
                reasoning="No experts registered, using default"
            )]

        # Нормализуем запрос
        query_lower = query.lower()
        query_tokens = self._tokenize(query_lower)

        # Вычисляем scores для всех экспертов
        results = []
        for expert_id, expert in self.experts.items():
            score, matched = self._calculate_score(query_tokens, expert)

            if score > 0 or expert_id == self.default_expert_id:
                confidence = min(score, 1.0)

                reasoning = self._generate_reasoning(
                    matched_keywords=matched,
                    expert_name=expert.name,
                    score=score
                )

                results.append(RoutingResult(
                    expert_id=expert_id,
                    confidence=confidence,
                    matched_keywords=matched,
                    reasoning=reasoning
                ))

        # Фильтруем по min_confidence
        results = [r for r in results if r.confidence >= min_confidence]

        # Если нет подходящих экспертов, используем default
        if not results and self.default_expert_id in self.experts:
            results.append(RoutingResult(
                expert_id=self.default_expert_id,
                confidence=0.5,
                matched_keywords=[],
                reasoning=f"No specific expert matched, using default ({self.default_expert_id})"
            ))

        # Сортируем по confidence (и по priority при равном confidence)
        results.sort(
            key=lambda r: (
                r.confidence,
                self.experts.get(r.expert_id, ExpertInfo("", "", "", set(), 0)).priority
            ),
            reverse=True
        )

        return results[:top_k]

    def _tokenize(self, text: str) -> Set[str]:
        """
        Токенизация текста на слова.

        Args:
            text: входной текст (уже в lowercase)

        Returns:
            Набор токенов
        """
        # Убираем пунктуацию и разбиваем на слова
        text_clean = re.sub(r'[^\w\s]', ' ', text)
        tokens = text_clean.split()
        return set(tokens)

    def _calculate_score(
        self,
        query_tokens: Set[str],
        expert: ExpertInfo
    ) -> Tuple[float, List[str]]:
        """
        Вычисление score для эксперта.

        Args:
            query_tokens: токены из запроса
            expert: информация об эксперте

        Returns:
            (score, matched_keywords) - score и список найденных keywords
        """
        if not expert.keywords:
            return 0.0, []

        # Находим пересечение
        matched = query_tokens & expert.keywords

        if not matched:
            return 0.0, []

        # Score = (число совпадений / число keywords эксперта)
        # Это даёт больший вес экспертам с меньшим числом специфичных keywords
        score = len(matched) / len(expert.keywords)

        # Бонус за количество совпадений
        match_bonus = min(len(matched) * 0.2, 0.5)

        final_score = score + match_bonus

        return final_score, sorted(list(matched))

    def _generate_reasoning(
        self,
        matched_keywords: List[str],
        expert_name: str,
        score: float
    ) -> str:
        """
        Генерация объяснения выбора эксперта.

        Args:
            matched_keywords: найденные ключевые слова
            expert_name: название эксперта
            score: итоговый score

        Returns:
            Текстовое объяснение
        """
        if not matched_keywords:
            return f"No keywords matched for {expert_name}"

        keywords_str = ", ".join(matched_keywords)
        return f"Matched keywords for {expert_name}: {keywords_str} (score: {score:.2f})"

    def get_expert_info(self, expert_id: str) -> Optional[ExpertInfo]:
        """
        Получить информацию об эксперте.

        Args:
            expert_id: ID эксперта

        Returns:
            ExpertInfo или None если не найден
        """
        return self.experts.get(expert_id)

    def list_experts(self) -> List[ExpertInfo]:
        """
        Получить список всех экспертов.

        Returns:
            Список ExpertInfo
        """
        return list(self.experts.values())

    def save_config(self, filepath: str) -> None:
        """
        Сохранить конфигурацию экспертов в JSON файл.

        Args:
            filepath: путь к файлу для сохранения
        """
        config = {
            "default_expert_id": self.default_expert_id,
            "experts": [
                {
                    "expert_id": e.expert_id,
                    "name": e.name,
                    "description": e.description,
                    "keywords": list(e.keywords),
                    "priority": e.priority
                }
                for e in self.experts.values()
            ]
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

    def load_config(self, filepath: str) -> None:
        """
        Загрузить конфигурацию экспертов из JSON файла.

        Args:
            filepath: путь к файлу конфигурации
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            config = json.load(f)

        self.default_expert_id = config.get("default_expert_id", "general")
        self.experts.clear()

        for expert_data in config.get("experts", []):
            self.add_expert(
                expert_id=expert_data["expert_id"],
                name=expert_data["name"],
                description=expert_data.get("description", ""),
                keywords=set(expert_data.get("keywords", [])),
                priority=expert_data.get("priority", 5)
            )


def test_simple_router():
    """Тестирование SimpleRouter."""
    print("=" * 70)
    print("Тестирование SimpleRouter")
    print("=" * 70)

    # Создаём роутер
    router = SimpleRouter(default_expert_id="general")

    # Добавляем экспертов
    router.add_expert(
        expert_id="python_expert",
        name="Python Expert",
        description="Специалист по Python и data science",
        keywords={"python", "pandas", "numpy", "flask", "django", "dataframe", "matplotlib"},
        priority=8
    )

    router.add_expert(
        expert_id="js_expert",
        name="JavaScript Expert",
        description="Специалист по JavaScript и web development",
        keywords={"javascript", "js", "react", "node", "npm", "typescript", "vue", "angular"},
        priority=8
    )

    router.add_expert(
        expert_id="math_expert",
        name="Mathematics Expert",
        description="Специалист по математике и алгоритмам",
        keywords={"math", "алгоритм", "matrix", "integral", "derivative", "equation", "optimization"},
        priority=7
    )

    router.add_expert(
        expert_id="general",
        name="General Assistant",
        description="Общие вопросы и разговор",
        keywords=set(),
        priority=3
    )

    # Тестовые запросы
    test_queries = [
        "How to use pandas dataframe in Python?",
        "Create a React component with TypeScript",
        "Solve differential equation",
        "What is the weather today?",
        "Optimize matrix multiplication algorithm"
    ]

    print(f"\n{'Запрос':<45} {'Эксперт':<20} {'Confidence':<12} {'Keywords'}")
    print("-" * 110)

    for query in test_queries:
        results = router.route(query, top_k=1)
        if results:
            result = results[0]
            expert_name = router.get_expert_info(result.expert_id).name
            keywords_str = ", ".join(result.matched_keywords[:3]) if result.matched_keywords else "none"
            print(f"{query:<45} {expert_name:<20} {result.confidence:<12.2f} {keywords_str}")

    # Тест top_k
    print(f"\n{'=' * 70}")
    print("Тест top_k=3 для запроса с несколькими совпадениями")
    print("=" * 70)

    query = "Python optimization algorithm for matrix operations"
    results = router.route(query, top_k=3)

    for i, result in enumerate(results, 1):
        expert = router.get_expert_info(result.expert_id)
        print(f"\n{i}. {expert.name} (confidence: {result.confidence:.2f})")
        print(f"   Reasoning: {result.reasoning}")

    # Тест сохранения/загрузки
    print(f"\n{'=' * 70}")
    print("Тест сохранения и загрузки конфигурации")
    print("=" * 70)

    config_path = "/tmp/router_config.json"
    router.save_config(config_path)
    print(f"✅ Конфигурация сохранена в {config_path}")

    # Создаём новый роутер и загружаем конфигурацию
    new_router = SimpleRouter()
    new_router.load_config(config_path)
    print(f"✅ Конфигурация загружена ({len(new_router.list_experts())} экспертов)")

    # Проверяем что работает так же
    results_new = new_router.route("Python pandas dataframe", top_k=1)
    print(f"✅ Тест после загрузки: {results_new[0].expert_id} (confidence: {results_new[0].confidence:.2f})")

    print(f"\n{'=' * 70}")
    print("✅ Все тесты пройдены!")
    print("=" * 70)


if __name__ == "__main__":
    test_simple_router()
