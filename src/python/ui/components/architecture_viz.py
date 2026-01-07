"""
Компонент Architecture Visualization для динамической визуализации архитектуры MoE системы.

Функциональность:
    - Интерактивный граф архитектуры (Plotly)
    - Показывает все модули (Router, Memory, Experts)
    - Количество слоёв и параметров каждого модуля
    - Real-time скорость обмена данными (анимированные рёбра)
    - Разные layouts (Hierarchical, Force-directed, Circular)
    - Auto-refresh для обновления data flow
"""

import gradio as gr
import plotly.graph_objects as go
import networkx as nx
import math
from typing import Dict, List, Tuple, Any


def create_architecture_viz(moe_system):
    """
    Создаёт интерфейс для визуализации архитектуры системы.

    Args:
        moe_system: Экземпляр MoESystem

    Returns:
        Gradio компоненты (размещённые внутри функции)
    """
    gr.Markdown("## Динамическая архитектура AI системы")
    gr.Markdown(
        "Визуализация всех модулей, их связей и потоков данных в реальном времени"
    )

    with gr.Row():
        # Главный граф архитектуры
        architecture_graph = gr.Plot(
            label="Architecture Graph",
            show_label=False
        )

    # Настройки визуализации
    with gr.Row():
        layout_dropdown = gr.Dropdown(
            choices=["Hierarchical", "Force-directed", "Circular"],
            value="Hierarchical",
            label="Layout алгоритм",
            info="Способ расположения узлов на графе",
            scale=1
        )

        refresh_btn = gr.Button("🔄 Refresh", scale=1, variant="primary")

        auto_refresh = gr.Checkbox(
            label="Auto-refresh (2s)",
            value=True,
            scale=1
        )

    # Детали модуля и Data Flow метрики
    with gr.Row():
        # Детали выбранного модуля
        with gr.Column(scale=1):
            gr.Markdown("### 📦 Детали модуля")

            module_name = gr.Textbox(
                label="Название",
                interactive=False,
                value="-"
            )

            module_type = gr.Textbox(
                label="Тип",
                interactive=False,
                value="-"
            )

            module_layers = gr.Number(
                label="Количество слоёв",
                interactive=False,
                value=0
            )

            module_params = gr.Number(
                label="Параметров",
                interactive=False,
                value=0
            )

            module_memory = gr.Number(
                label="Память (MB)",
                interactive=False,
                value=0.0,
                precision=2
            )

        # Data flow метрики
        with gr.Column(scale=2):
            gr.Markdown("### 🔄 Data Flow Metrics")

            dataflow_table = gr.Dataframe(
                headers=["From → To", "Speed (KB/s)", "Total Transferred (MB)", "Activity"],
                datatype=["str", "number", "number", "str"],
                interactive=False,
                wrap=True
            )

    # === 3-уровневая система памяти ===
    gr.Markdown("## 🧠 Трёхуровневая система памяти")
    gr.Markdown(
        "Визуализация всех трёх уровней памяти: Current (детальный), Obsolete (сжатый), LongTerm (резюме)"
    )

    with gr.Row():
        # Уровень 1: Current Memory
        with gr.Column(scale=1):
            gr.Markdown("### 💾 Current Memory")
            gr.Markdown("*Полный детальный контекст (250k токенов)*")

            current_chunks = gr.Number(
                label="Chunks",
                interactive=False,
                value=0
            )

            current_tokens = gr.Number(
                label="Tokens",
                interactive=False,
                value=0
            )

            current_usage = gr.Number(
                label="Usage (%)",
                interactive=False,
                value=0.0,
                precision=1
            )

            current_content = gr.Dataframe(
                headers=["Content", "Tokens", "Importance"],
                datatype=["str", "number", "number"],
                interactive=False,
                label="Последние сообщения",
                wrap=True
            )

        # Уровень 2: Obsolete Memory
        with gr.Column(scale=1):
            gr.Markdown("### 📦 Obsolete Memory")
            gr.Markdown("*Сжатая информация (250k токенов)*")

            obsolete_chunks = gr.Number(
                label="Chunks",
                interactive=False,
                value=0
            )

            obsolete_tokens = gr.Number(
                label="Tokens",
                interactive=False,
                value=0
            )

            obsolete_usage = gr.Number(
                label="Usage (%)",
                interactive=False,
                value=0.0,
                precision=1
            )

            obsolete_content = gr.Dataframe(
                headers=["Content", "Tokens", "Compressed"],
                datatype=["str", "number", "bool"],
                interactive=False,
                label="Сжатые куски",
                wrap=True
            )

        # Уровень 3: LongTerm Memory
        with gr.Column(scale=1):
            gr.Markdown("### 🗄️ LongTerm Memory")
            gr.Markdown("*Краткие резюме (250k токенов)*")

            longterm_chunks = gr.Number(
                label="Chunks",
                interactive=False,
                value=0
            )

            longterm_tokens = gr.Number(
                label="Tokens",
                interactive=False,
                value=0
            )

            longterm_usage = gr.Number(
                label="Usage (%)",
                interactive=False,
                value=0.0,
                precision=1
            )

            longterm_content = gr.Dataframe(
                headers=["Content", "Tokens", "Compressed"],
                datatype=["str", "number", "bool"],
                interactive=False,
                label="Ультра-сжатые резюме",
                wrap=True
            )

    # Общая статистика
    with gr.Row():
        total_memory_tokens = gr.Number(
            label="Общее количество токенов во всех уровнях",
            interactive=False,
            value=0,
            scale=2
        )

        memory_refresh_btn = gr.Button("🔄 Обновить память", scale=1, variant="secondary")

    # Легенда
    gr.Markdown(
        """
        ### 📖 Легенда

        **Узлы (модули):**
        - 🔴 **Router** - маршрутизация запросов к экспертам
        - 🔵 **Memory** - трёхуровневая система памяти
        - 🟢 **Expert** - domain-specific эксперты
        - 🟡 **Trainer** - обучение моделей

        **Рёбра (связи):**
        - Толщина ~ скорость передачи данных (KB/s)
        - Прозрачность ~ уровень активности соединения
        - Hover для просмотра деталей
        """
    )

    # === Helper Functions ===

    def get_module_color(module_type: str) -> str:
        """Возвращает цвет узла по типу модуля."""
        colors = {
            'router': '#FF6B6B',       # Красный
            'memory': '#4ECDC4',       # Бирюзовый
            'expert': '#95E1D3',       # Светло-зелёный
            'trainer': '#F38181',      # Розовый
            'transformer': '#AA96DA'   # Фиолетовый
        }
        return colors.get(module_type, '#CCCCCC')

    def calculate_node_size(params_count: int) -> int:
        """Размер узла пропорционален количеству параметров."""
        if params_count == 0:
            return 30

        # Логарифмическая шкала
        size = min(30 + math.log10(params_count + 1) * 15, 100)
        return int(size)

    def calculate_edge_width(data_flow: float) -> float:
        """Толщина ребра пропорциональна data flow."""
        if data_flow == 0:
            return 2.0

        # Логарифмическая шкала
        width = min(2.0 + math.log10(data_flow + 1) * 2, 15.0)
        return width

    def calculate_positions(
        nodes: List[Dict],
        edges: List[Dict],
        layout: str
    ) -> Dict[str, Tuple[float, float]]:
        """
        Вычисляет позиции узлов в зависимости от layout.

        Args:
            nodes: Список узлов
            edges: Список рёбер
            layout: Тип layout ('Hierarchical', 'Force-directed', 'Circular')

        Returns:
            Словарь {node_id: (x, y)}
        """
        if layout == "Hierarchical":
            # Уровни: Router/Memory (top) → Experts (middle) → Output (bottom)
            positions = {}

            # Уровень 1: Router и Memory
            positions['Router'] = (0, 3)
            positions['Memory'] = (2, 3)

            # Уровень 2: Experts (горизонтально)
            expert_nodes = [n for n in nodes if n['type'] == 'expert']

            if expert_nodes:
                experts_count = len(expert_nodes)
                spacing = 2.0

                for i, expert in enumerate(expert_nodes):
                    x = -spacing * (experts_count - 1) / 2 + i * spacing
                    positions[expert['id']] = (x, 1.5)

            # Уровень 3: Trainer (если есть)
            if any(n['id'] == 'Trainer' for n in nodes):
                positions['Trainer'] = (1, 0)

            return positions

        elif layout == "Force-directed":
            # Используем NetworkX для force-directed layout
            G = nx.Graph()

            for node in nodes:
                G.add_node(node['id'])

            for edge in edges:
                G.add_edge(edge['from'], edge['to'])

            try:
                pos = nx.spring_layout(G, seed=42, k=1.5, iterations=50)

                # Масштабируем координаты
                positions = {
                    node_id: (x * 5, y * 5)
                    for node_id, (x, y) in pos.items()
                }
                return positions

            except Exception:
                # Fallback на hierarchical
                return calculate_positions(nodes, edges, "Hierarchical")

        elif layout == "Circular":
            # Круговое расположение
            positions = {}
            n = len(nodes)

            for i, node in enumerate(nodes):
                angle = 2 * math.pi * i / n
                x = 3 * math.cos(angle)
                y = 3 * math.sin(angle)
                positions[node['id']] = (x, y)

            return positions

        else:
            # Default: Hierarchical
            return calculate_positions(nodes, edges, "Hierarchical")

    def build_architecture_graph(layout_type: str) -> go.Figure:
        """
        Построение интерактивного графа архитектуры.

        Args:
            layout_type: Тип layout ('Hierarchical', 'Force-directed', 'Circular')

        Returns:
            Plotly Figure
        """
        try:
            # Получаем данные архитектуры от backend
            arch_data = moe_system.get_architecture_graph()

            nodes = arch_data['nodes']
            edges = arch_data['edges']

            if not nodes:
                # Пустой граф
                fig = go.Figure()
                fig.add_annotation(
                    text="Нет доступных модулей для визуализации",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(size=16)
                )
                return fig

            # Вычисляем позиции узлов
            node_positions = calculate_positions(nodes, edges, layout_type)

            # Создаём Figure
            fig = go.Figure()

            # === Добавляем рёбра (edges) ===
            for edge in edges:
                from_id = edge['from']
                to_id = edge['to']

                if from_id not in node_positions or to_id not in node_positions:
                    continue

                from_pos = node_positions[from_id]
                to_pos = node_positions[to_id]

                # Толщина и прозрачность
                data_flow = edge.get('data_flow', 0.0)
                activity_level = edge.get('activity_level', 0.1)

                line_width = calculate_edge_width(data_flow)
                line_opacity = 0.3 + (0.7 * activity_level)

                # Добавляем линию
                fig.add_trace(go.Scatter(
                    x=[from_pos[0], to_pos[0]],
                    y=[from_pos[1], to_pos[1]],
                    mode='lines',
                    line=dict(
                        width=line_width,
                        color=f'rgba(100, 150, 255, {line_opacity})'
                    ),
                    hovertemplate=(
                        f"<b>{from_id} → {to_id}</b><br>"
                        f"Data Flow: {data_flow / 1024:.2f} KB/s<br>"
                        f"Activity: {activity_level * 100:.1f}%<br>"
                        "<extra></extra>"
                    ),
                    showlegend=False
                ))

            # === Добавляем узлы (nodes) ===
            for node in nodes:
                node_id = node['id']

                if node_id not in node_positions:
                    continue

                pos = node_positions[node_id]

                # Размер и цвет узла
                node_size = calculate_node_size(node.get('params_count', 0))
                node_color = get_module_color(node.get('type', 'unknown'))

                # Добавляем узел
                fig.add_trace(go.Scatter(
                    x=[pos[0]],
                    y=[pos[1]],
                    mode='markers+text',
                    marker=dict(
                        size=node_size,
                        color=node_color,
                        line=dict(width=3, color='white'),
                        opacity=0.9
                    ),
                    text=node_id,
                    textposition='bottom center',
                    textfont=dict(size=12, color='black', family='Arial Black'),
                    name=node_id,
                    hovertemplate=(
                        f"<b>{node_id}</b><br>"
                        f"Type: {node.get('type', 'unknown')}<br>"
                        f"Layers: {node.get('layers', 0)}<br>"
                        f"Parameters: {node.get('params_count', 0):,}<br>"
                        f"Memory: {node.get('memory_mb', 0.0):.1f} MB<br>"
                        "<extra></extra>"
                    ),
                    showlegend=False
                ))

            # Настройка layout
            fig.update_layout(
                title=dict(
                    text="MoE System Architecture (Real-Time)",
                    font=dict(size=20, color='#333')
                ),
                showlegend=False,
                hovermode='closest',
                xaxis=dict(
                    showgrid=False,
                    zeroline=False,
                    showticklabels=False,
                    title=""
                ),
                yaxis=dict(
                    showgrid=False,
                    zeroline=False,
                    showticklabels=False,
                    title=""
                ),
                plot_bgcolor='rgba(240, 245, 250, 0.8)',
                paper_bgcolor='white',
                width=1200,
                height=700,
                margin=dict(l=40, r=40, t=60, b=40)
            )

            return fig

        except Exception as e:
            # Возвращаем пустой граф с ошибкой
            fig = go.Figure()
            fig.add_annotation(
                text=f"Ошибка при построении графа: {str(e)}",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=14, color='red')
            )
            return fig

    def get_dataflow_stats() -> List[List]:
        """Возвращает статистику data flow для таблицы."""
        try:
            stats = moe_system.get_data_flow_stats()

            rows = []
            for (from_module, to_module), speed_bytes in stats.items():
                speed_kb = speed_bytes / 1024

                # Примерный total transferred (упрощённая оценка)
                total_mb = speed_kb * 60 / 1024  # За последние 60 секунд

                # Activity level
                if speed_kb > 100:
                    activity = "🟢 High"
                elif speed_kb > 10:
                    activity = "🟡 Medium"
                else:
                    activity = "🔵 Low"

                rows.append([
                    f"{from_module} → {to_module}",
                    round(speed_kb, 2),
                    round(total_mb, 2),
                    activity
                ])

            return rows

        except Exception:
            return []

    def get_memory_visualization():
        """Возвращает данные о памяти для визуализации."""
        try:
            # Получаем статистику
            stats = moe_system.get_memory_stats()

            # Получаем содержимое
            content = moe_system.get_memory_content(max_items_per_level=5)

            # Форматируем данные для таблиц
            current_rows = [
                [item['content'], item['tokens'], item['importance']]
                for item in content['current']
            ]

            obsolete_rows = [
                [item['content'], item['tokens'], item['compressed']]
                for item in content['obsolete']
            ]

            longterm_rows = [
                [item['content'], item['tokens'], item['compressed']]
                for item in content['longterm']
            ]

            return (
                stats['current']['chunks'],
                stats['current']['tokens'],
                stats['current']['usage_pct'],
                current_rows,
                stats['obsolete']['chunks'],
                stats['obsolete']['tokens'],
                stats['obsolete']['usage_pct'],
                obsolete_rows,
                stats['longterm']['chunks'],
                stats['longterm']['tokens'],
                stats['longterm']['usage_pct'],
                longterm_rows,
                stats['total_tokens']
            )

        except Exception as e:
            print(f"Error getting memory visualization: {e}")
            # Возвращаем пустые значения
            return (0, 0, 0.0, [], 0, 0, 0.0, [], 0, 0, 0.0, [], 0)

    # === Event Handlers ===

    def update_visualization(layout_type: str):
        """Обновляет визуализацию и data flow таблицу."""
        graph = build_architecture_graph(layout_type)
        dataflow = get_dataflow_stats()

        return graph, dataflow

    # Ручное обновление
    refresh_btn.click(
        fn=update_visualization,
        inputs=[layout_dropdown],
        outputs=[architecture_graph, dataflow_table]
    )

    # Изменение layout
    layout_dropdown.change(
        fn=update_visualization,
        inputs=[layout_dropdown],
        outputs=[architecture_graph, dataflow_table]
    )

    # Обновление памяти
    memory_refresh_btn.click(
        fn=get_memory_visualization,
        outputs=[
            current_chunks,
            current_tokens,
            current_usage,
            current_content,
            obsolete_chunks,
            obsolete_tokens,
            obsolete_usage,
            obsolete_content,
            longterm_chunks,
            longterm_tokens,
            longterm_usage,
            longterm_content,
            total_memory_tokens
        ]
    )

    # Auto-refresh
    timer = gr.Timer(2.0)

    def maybe_refresh(auto_enabled: bool, layout_type: str):
        """Обновляет граф только если auto-refresh включён."""
        if auto_enabled:
            graph, dataflow = update_visualization(layout_type)
            memory_data = get_memory_visualization()
            return (graph, dataflow) + memory_data
        else:
            return (gr.update(), gr.update()) + tuple([gr.update()] * 13)

    timer.tick(
        fn=maybe_refresh,
        inputs=[auto_refresh, layout_dropdown],
        outputs=[
            architecture_graph,
            dataflow_table,
            current_chunks,
            current_tokens,
            current_usage,
            current_content,
            obsolete_chunks,
            obsolete_tokens,
            obsolete_usage,
            obsolete_content,
            longterm_chunks,
            longterm_tokens,
            longterm_usage,
            longterm_content,
            total_memory_tokens
        ]
    )

    # Инициализация при загрузке
    initial_graph = build_architecture_graph("Hierarchical")
    initial_dataflow = get_dataflow_stats()
    initial_memory = get_memory_visualization()

    architecture_graph.value = initial_graph
    dataflow_table.value = initial_dataflow

    # Инициализация памяти
    (
        current_chunks.value,
        current_tokens.value,
        current_usage.value,
        current_content.value,
        obsolete_chunks.value,
        obsolete_tokens.value,
        obsolete_usage.value,
        obsolete_content.value,
        longterm_chunks.value,
        longterm_tokens.value,
        longterm_usage.value,
        longterm_content.value,
        total_memory_tokens.value
    ) = initial_memory
