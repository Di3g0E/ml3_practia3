# Práctica 3: Aprendizaje por Refuerzo (Reinforcement Learning)

Este proyecto implementa y compara dos algoritmos fundamentales de Aprendizaje por Refuerzo: **REINFORCE** y **Actor-Critic**. Los agentes son entrenados y evaluados en entornos de Gymnasium como `CartPole-v1`.

## Requisitos

El proyecto requiere Python 3.x y las siguientes librerías, listadas en `requirements.txt`:

*   `torch`: Framework de Deep Learning.
*   `gymnasium`: Entorno de simulación para RL (sucesor de OpenAI Gym).
*   `pygame`: Necesario para el renderizado de algunos entornos de Gymnasium (ej. CartPole).
*   `numpy`: Computación numérica.
*   `matplotlib`: Generación de gráficas de resultados.
*   `moviepy`: (Opcional) Para la generación y edición de videos de los agentes.

### Instalación

Para instalar todas las dependencias necesarias, ejecuta el siguiente comando en tu terminal:

```bash
pip install -r requirements.txt
```

## Estructura del Proyecto

*   `src/`: Código fuente de los agentes y utilidades.
    *   `src/agents/reinforce.py`: Implementación del algoritmo REINFORCE.
    *   `src/agents/actor_critic.py`: Implementación del algoritmo Actor-Critic.
    *   `src/utils/`: Funciones de utilidad (ej. ploteo).
*   `models/`: Directorio donde se guardan los modelos entrenados (`.pth`).
*   `results/`: Directorio donde se guardan los resultados de entrenamiento (returns) y las gráficas generadas.
*   `videos/`: Directorio donde se guardan las grabaciones de los agentes interactuando con el entorno.
*   `rl_main.py`: Script principal para entrenar y evaluar los agentes.
*   `test_agents.py`: Script básico para verificar que los agentes se inicializan y ejecutan correctamente (unit tests simples).

## Ejecución

El script principal es `rl_main.py`. Permite entrenar un agente específico en un entorno determinado.

### Comando Básico

```bash
python rl_main.py <nombre_entorno> <nombre_agente> [opciones]
```

### Argumentos

*   `env_name` (requerido): Nombre del entorno de Gymnasium (ej. `CartPole-v1`).
*   `agent_name` (requerido): Algoritmo a utilizar. Opciones: `reinforce`, `actorcritic`.
*   `--n_episodes` (opcional): Número de episodios de entrenamiento. Si no se especifica, usa un valor por defecto (1000 para CartPole, 2000 para otros).

### Ejemplos de Uso

1.  **Entrenar REINFORCE en CartPole-v1:**
    ```bash
    python rl_main.py CartPole-v1 reinforce
    ```

2.  **Entrenar Actor-Critic en CartPole-v1 con 500 episodios:**
    ```bash
    python rl_main.py CartPole-v1 actorcritic --n_episodes 500
    ```

## Resultados y Conclusiones

Tras la ejecución, el sistema generará varios archivos en la carpeta `results/`:
*   Archivos `.npy` con los retornos obtenidos en cada episodio.
*   Imágenes `.png` mostrando la curva de aprendizaje (retorno vs episodios) y las pérdidas (loss).

### Comparativa: REINFORCE vs Actor-Critic

Basado en la teoría y experimentos típicos en entornos como CartPole:

1.  **Estabilidad y Varianza**:
    *   **REINFORCE** (Monte Carlo Policy Gradient) utiliza el retorno completo del episodio para actualizar la política. Esto introduce una **alta varianza**, ya que los retornos pueden variar mucho de un episodio a otro, haciendo el aprendizaje más inestable y ruidoso.
    *   **Actor-Critic** reduce esta varianza utilizando un "Crítico" que estima el valor del estado (Baseline). La actualización se hace paso a paso (TD Error) o usando la ventaja (Advantage), lo que generalmente resulta en un entrenamiento **más estable** y una convergencia más suave.

2.  **Velocidad de Convergencia**:
    *   **Actor-Critic** suele converger más rápido en términos de "muestras eficientes" debido a la menor varianza en las actualizaciones de gradiente. REINFORCE puede requerir más episodios para promediar el ruido y encontrar una buena política.

3.  **Complejidad**:
    *   REINFORCE es más sencillo de implementar. Actor-Critic añade complejidad al requerir dos redes neuronales (Actor y Crítico) que deben entrenarse simultáneamente.

### Gráficas Generadas
Las gráficas en `results/` permiten visualizar estas diferencias. Se espera observar que la curva de retorno de Actor-Critic sea menos ruidosa y ascienda de manera más consistente que la de REINFORCE.

## Experimentos y Ajuste de Hiperparámetros

Se ha realizado una serie de experimentos variando la tasa de aprendizaje (Learning Rate) y el factor de descuento (Gamma) en el entorno `CartPole-v1` durante 200 episodios.

### Tabla de Resultados (Promedio últimos 50 episodios)

| Identificador del Experimento | Mean Return (Last 50) | Max Return |
| :--- | :--- | :--- |
| `reinforce_lr0.001_g0.99` | **135.70** | **373.00** |
| `reinforce_lr0.01_g0.99` | 9.44 | 33.00 |
| `actorcritic_lr0.001_g0.99` | 9.80 | 47.00 |
| `actorcritic_lr0.01_g0.99` | 9.24 | 27.00 |
| `actorcritic_lr0.001_g0.95` | 10.36 | 88.00 |

### Análisis

1.  **Sensibilidad al Learning Rate**:
    *   **REINFORCE** mostró un desempeño superior con una tasa de aprendizaje de `0.001`. Al aumentar la tasa a `0.01`, el agente no logró aprender (retorno cercano al mínimo de 9-10), lo que confirma la inestabilidad de Policy Gradient con pasos de actualización demasiado grandes.
    
2.  **Desempeño de Actor-Critic**:
    *   En este breve conjunto de pruebas (200 episodios), el agente **Actor-Critic** no logró superar a REINFORCE. Esto puede deberse a que Actor-Critic suele requerir un ajuste más fino de hiperparámetros (balance entre LR de actor y crítico) o un mayor número de episodios para estabilizar el entrenamiento de ambas redes simultáneamente.

3.  **Conclusión**:
    *   Para `CartPole-v1` con un presupuesto de episodios limitado, **REINFORCE con lr=0.001** resultó ser la configuración más eficiente. Actor-Critic podría requerir `target_update_freq` diferente o más exploración inicial para obtener mejores resultados.
