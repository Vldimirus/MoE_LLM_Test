"""
Примеры использования SimpleRouter.

Демонстрирует основные возможности роутера.
"""

import sys
import os

# Добавляем путь к src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from python.routing.router import SimpleRouter


def example_1_basic_routing():
    """Пример 1: Базовая маршрутизация."""
    print("=" * 70)
    print("Пример 1: Базовая маршрутизация")
    print("=" * 70)

    # Создаём роутер
    router = SimpleRouter(default_expert_id="general")

    # Добавляем экспертов
    router.add_expert(
        expert_id="python",
        name="Python Expert",
        keywords={"python", "pandas", "numpy", "flask"},
        priority=8
    )

    router.add_expert(
        expert_id="javascript",
        name="JavaScript Expert",
        keywords={"javascript", "react", "node", "typescript"},
        priority=8
    )

    router.add_expert(
        expert_id="general",
        name="General Assistant",
        keywords=set(),
        priority=3
    )

    # Тестовые запросы
    queries = [
        "How to use pandas in Python?",
        "Create React component",
        "What's the weather?",
    ]

    print(f"\n{'Query':<35} {'Expert':<20} {'Confidence':<12}")
    print("-" * 70)

    for query in queries:
        results = router.route(query)
        result = results[0]
        expert = router.get_expert_info(result.expert_id)
        print(f"{query:<35} {expert.name:<20} {result.confidence:.2f}")


def example_2_top_k_routing():
    """Пример 2: Top-K маршрутизация."""
    print("\n" + "=" * 70)
    print("Пример 2: Top-K маршрутизация")
    print("=" * 70)

    router = SimpleRouter()

    # Добавляем экспертов
    router.add_expert(
        expert_id="python",
        name="Python Expert",
        keywords={"python", "pandas", "numpy", "scipy", "sklearn"},
        priority=9
    )

    router.add_expert(
        expert_id="ml",
        name="Machine Learning Expert",
        keywords={"ml", "model", "training", "sklearn", "pytorch"},
        priority=8
    )

    router.add_expert(
        expert_id="data_science",
        name="Data Science Expert",
        keywords={"data", "analysis", "visualization", "pandas", "matplotlib"},
        priority=7
    )

    # Запрос с несколькими совпадениями
    query = "Train ML model with sklearn and pandas in Python"

    print(f"\nQuery: {query}")
    print(f"\nTop-3 Experts:\n")

    results = router.route(query, top_k=3)

    for i, result in enumerate(results, 1):
        expert = router.get_expert_info(result.expert_id)
        print(f"{i}. {expert.name}")
        print(f"   Confidence: {result.confidence:.2f}")
        print(f"   Matched keywords: {', '.join(result.matched_keywords)}")
        print(f"   {result.reasoning}\n")


def example_3_confidence_filtering():
    """Пример 3: Фильтрация по confidence."""
    print("=" * 70)
    print("Пример 3: Фильтрация по минимальной уверенности")
    print("=" * 70)

    router = SimpleRouter()

    router.add_expert(
        expert_id="python",
        name="Python Expert",
        keywords={"python", "flask", "django"},
        priority=8
    )

    router.add_expert(
        expert_id="web",
        name="Web Development Expert",
        keywords={"web", "html", "css", "javascript", "frontend", "backend"},
        priority=7
    )

    query = "Build web application with Flask"

    # Разные уровни min_confidence
    confidence_levels = [0.0, 0.3, 0.5, 0.7]

    print(f"\nQuery: {query}\n")

    for min_conf in confidence_levels:
        results = router.route(query, top_k=5, min_confidence=min_conf)
        print(f"min_confidence={min_conf:.1f}: {len(results)} experts matched")
        for result in results:
            expert = router.get_expert_info(result.expert_id)
            print(f"  - {expert.name}: {result.confidence:.2f}")
        print()


def example_4_save_load_config():
    """Пример 4: Сохранение и загрузка конфигурации."""
    print("=" * 70)
    print("Пример 4: Сохранение и загрузка конфигурации")
    print("=" * 70)

    # Создаём роутер с конфигурацией
    router = SimpleRouter(default_expert_id="general")

    # Добавляем экспертов для программирования
    experts_config = [
        {
            "expert_id": "python",
            "name": "Python Expert",
            "keywords": {"python", "pandas", "numpy", "flask", "django", "pytorch"},
            "priority": 9
        },
        {
            "expert_id": "javascript",
            "name": "JavaScript Expert",
            "keywords": {"javascript", "js", "react", "vue", "node", "npm", "typescript"},
            "priority": 9
        },
        {
            "expert_id": "rust",
            "name": "Rust Expert",
            "keywords": {"rust", "cargo", "tokio", "async"},
            "priority": 8
        },
        {
            "expert_id": "cpp",
            "name": "C++ Expert",
            "keywords": {"cpp", "c++", "stl", "template"},
            "priority": 8
        },
        {
            "expert_id": "general",
            "name": "General Programming",
            "keywords": set(),
            "priority": 5
        }
    ]

    for config in experts_config:
        router.add_expert(**config)

    print(f"\n✅ Создан роутер с {len(router.list_experts())} экспертами\n")

    # Сохраняем конфигурацию
    config_path = "/tmp/programming_router_config.json"
    router.save_config(config_path)
    print(f"✅ Конфигурация сохранена в {config_path}\n")

    # Тест оригинального роутера
    test_query = "How to use async/await in Rust?"
    result_original = router.route(test_query)[0]
    print(f"Оригинальный роутер: {result_original.expert_id} (confidence: {result_original.confidence:.2f})")

    # Создаём новый роутер и загружаем конфигурацию
    new_router = SimpleRouter()
    new_router.load_config(config_path)
    print(f"✅ Конфигурация загружена в новый роутер ({len(new_router.list_experts())} экспертов)\n")

    # Тест загруженного роутера
    result_loaded = new_router.route(test_query)[0]
    print(f"Загруженный роутер: {result_loaded.expert_id} (confidence: {result_loaded.confidence:.2f})")

    # Проверка идентичности
    if result_original.expert_id == result_loaded.expert_id:
        print("\n✅ Результаты идентичны - конфигурация загружена корректно!")
    else:
        print("\n❌ Результаты различаются")


def example_5_priority_system():
    """Пример 5: Система приоритетов."""
    print("\n" + "=" * 70)
    print("Пример 5: Система приоритетов экспертов")
    print("=" * 70)

    router = SimpleRouter()

    # Два эксперта с одинаковыми keywords, но разным priority
    router.add_expert(
        expert_id="general_web",
        name="General Web Developer",
        keywords={"web", "frontend"},
        priority=5  # Низкий приоритет
    )

    router.add_expert(
        expert_id="frontend_specialist",
        name="Frontend Specialist",
        keywords={"web", "frontend"},
        priority=9  # Высокий приоритет
    )

    query = "Build frontend web application"

    print(f"\nQuery: {query}")
    print(f"\nOба эксперта имеют одинаковые keywords: {{web, frontend}}")
    print(f"Но разные приоритеты: General Web (5) vs Frontend Specialist (9)\n")

    results = router.route(query, top_k=2)

    print("Результаты (сортировка по confidence, затем по priority):\n")
    for i, result in enumerate(results, 1):
        expert = router.get_expert_info(result.expert_id)
        print(f"{i}. {expert.name}")
        print(f"   Priority: {expert.priority}")
        print(f"   Confidence: {result.confidence:.2f}")
        print()


def example_6_expert_management():
    """Пример 6: Управление экспертами."""
    print("=" * 70)
    print("Пример 6: Управление экспертами")
    print("=" * 70)

    router = SimpleRouter()

    # Добавление экспертов
    print("\n1. Добавление экспертов:")
    router.add_expert("python", "Python Expert", keywords={"python"}, priority=8)
    router.add_expert("javascript", "JS Expert", keywords={"js"}, priority=8)
    router.add_expert("rust", "Rust Expert", keywords={"rust"}, priority=7)

    print(f"   Всего экспертов: {len(router.list_experts())}")

    # Список экспертов
    print("\n2. Список экспертов:")
    for expert in router.list_experts():
        print(f"   - {expert.name} ({expert.expert_id}) - priority: {expert.priority}")

    # Информация об эксперте
    print("\n3. Информация об эксперте 'python':")
    expert = router.get_expert_info("python")
    if expert:
        print(f"   Name: {expert.name}")
        print(f"   Description: {expert.description or 'N/A'}")
        print(f"   Keywords: {expert.keywords}")
        print(f"   Priority: {expert.priority}")

    # Удаление эксперта
    print("\n4. Удаление эксперта 'rust':")
    success = router.remove_expert("rust")
    print(f"   Успешно: {success}")
    print(f"   Осталось экспертов: {len(router.list_experts())}")

    # Попытка удалить несуществующего эксперта
    print("\n5. Попытка удалить несуществующего эксперта:")
    success = router.remove_expert("golang")
    print(f"   Успешно: {success} (expected False)")


def example_7_real_world_scenario():
    """Пример 7: Реальный сценарий использования."""
    print("\n" + "=" * 70)
    print("Пример 7: Реальный сценарий - MoE система для разработки")
    print("=" * 70)

    # Создаём полноценную конфигурацию для programming assistant
    router = SimpleRouter(default_expert_id="general")

    # Programming experts
    router.add_expert(
        "python", "Python Expert",
        keywords={"python", "pandas", "numpy", "flask", "django", "pytorch", "tensorflow"},
        priority=9
    )

    router.add_expert(
        "javascript", "JavaScript Expert",
        keywords={"javascript", "js", "react", "vue", "angular", "node", "npm", "typescript"},
        priority=9
    )

    router.add_expert(
        "systems", "Systems Programming Expert",
        keywords={"rust", "c", "cpp", "c++", "go", "memory", "performance"},
        priority=8
    )

    router.add_expert(
        "database", "Database Expert",
        keywords={"sql", "database", "postgresql", "mysql", "mongodb", "redis"},
        priority=8
    )

    router.add_expert(
        "devops", "DevOps Expert",
        keywords={"docker", "kubernetes", "ci", "cd", "aws", "deployment", "nginx"},
        priority=8
    )

    router.add_expert(
        "ml", "Machine Learning Expert",
        keywords={"ml", "ai", "model", "training", "neural", "deep learning"},
        priority=9
    )

    router.add_expert(
        "general", "General Programming",
        keywords=set(),
        priority=5
    )

    # Реальные запросы
    real_queries = [
        "How to deploy Flask app to AWS with Docker?",
        "Optimize SQL query performance in PostgreSQL",
        "Train neural network with PyTorch",
        "Build React frontend with TypeScript",
        "Implement memory-safe linked list in Rust",
        "Set up CI/CD pipeline with GitHub Actions",
        "What is the best programming language?",
    ]

    print(f"\n{'Query':<50} {'Expert':<25} {'Conf':<6} {'Keywords'}")
    print("-" * 110)

    for query in real_queries:
        results = router.route(query, top_k=1)
        result = results[0]
        expert = router.get_expert_info(result.expert_id)
        keywords_str = ", ".join(result.matched_keywords[:3]) if result.matched_keywords else "none"
        print(f"{query:<50} {expert.name:<25} {result.confidence:<6.2f} {keywords_str}")

    # Сохраняем эту конфигурацию для дальнейшего использования
    router.save_config("/tmp/moe_programming_config.json")
    print(f"\n✅ Конфигурация MoE системы сохранена в /tmp/moe_programming_config.json")


if __name__ == "__main__":
    example_1_basic_routing()
    example_2_top_k_routing()
    example_3_confidence_filtering()
    example_4_save_load_config()
    example_5_priority_system()
    example_6_expert_management()
    example_7_real_world_scenario()

    print("\n" + "=" * 70)
    print("✅ Все примеры выполнены успешно!")
    print("=" * 70)
