# Práctica 3: Aprendizaje por Refuerzo (Reinforcement Learning)

Este proyecto implementa, entrena y compara dos algoritmos fundamentales de Aprendizaje por Refuerzo: **REINFORCE** y **Actor-Critic**. Los agentes son evaluados en entornos de Gymnasium como `CartPole-v1` y `LunarLander-v3`.

## Requisitos

El proyecto requiere Python 3.x y las siguientes librerías principales (ver `requirements.txt`):

*   `torch`: Framework de Deep Learning para las redes neuronales.
*   `gymnasium`: Entorno de simulación (sucesor de OpenAI Gym).
*   `gymnasium[box2d]`: Necesario para el entorno LunarLander.
*   `pygame`: Necesario para el renderizado.
*   `numpy`: Computación numérica.
*   `matplotlib`: Generación de gráficas de resultados.
*   `moviepy`: Para la generación y edición de videos de los agentes.

### Instalación

Para instalar todas las dependencias necesarias, ejecuta:

```bash
pip install -r requirements.txt
```

## Estructura del Proyecto

*   `src/`: Código fuente.
    *   `src/agents/reinforce.py`: Implementación del algoritmo REINFORCE.
    *   `src/agents/actor_critic.py`: Implementación de Actor-Critic (con soporte para Dueling y Double DQN en el Crítico).
    *   `src/utils/`: Utilidades para ploteo y manejo de datos.
*   `experiments/`: Scripts de automatización de pruebas.
    *   `experiments/comparation_results.py`: Script principal que ejecuta la comparativa completa (Reinforce vs Actor-Critic) en todos los entornos.
*   `models/`: Almacena los modelos entrenados (`.pth`).
*   `results/`: Almacena gráficas de pérdidas/retornos (`.png`) y datos crudos (`.npy`).
*   `videos/`: Grabaciones de los agentes interactuando con el entorno.
*   `rl_main.py`: Script principal para entrenamiento individual.

## Ejecución

### 1. Entrenamiento Individual (`rl_main.py`)

Usa este script para entrenar un agente específico con una configuración personalizada.

**Sintaxis:**
```bash
python rl_main.py <env_name> <agent_name> [opciones]
```

**Argumentos Principales:**
*   `env_name`: Entorno a utilizar (`CartPole-v1`, `LunarLander-v3`).
*   `agent_name`: Agente a entrenar (`reinforce`, `actorcritic`).
*   `--n_episodes N`: Número de episodios de entrenamiento.
*   `--lr LR`: Tasa de aprendizaje (default: 0.001).
*   `--gamma G`: Factor de descuento (default: 0.99).

**Opciones Avanzadas (Actor-Critic):**
*   `--lr_critic LR`: Tasa de aprendizaje específica para el crítico.
*   `--entropy_coef C`: Coeficiente de regularización por entropía (ayuda a la exploración).
*   `--dueling`: Activa la arquitectura Dueling DQN en el crítico.
*   `--no_double`: Desactiva Double DQN (por defecto activo si se usa Dueling).

**Ejemplos:**

*   Entrenar REINFORCE en CartPole:
    ```bash
    python rl_main.py CartPole-v1 reinforce
    ```

*   Entrenar Actor-Critic en LunarLander con configuración avanzada:
    ```bash
    python rl_main.py LunarLander-v3 actorcritic --n_episodes 2000 --dueling --entropy_coef 0.01
    ```

### 2. Ejecutar Comparativa Completa

Para reproducir los resultados de la práctica (entrenar ambos agentes en ambos entornos y generar gráficas comparativas), ejecuta:

```bash
python experiments/comparation_results.py
```

*   Este script generará automáticamente archivos `comparison_*.png` en la carpeta `results/` mostrando las curvas de aprendizaje enfrentadas y una gráfica de barras con el desempeño en test.
*   Usa la opción `--quick` para una prueba rápida de funcionamiento (pocos episodios).

## Resultados Esperados

Los archivos generados en `results/` permitirán analizar:
1.  **Estabilidad**: Comparar la varianza de REINFORCE vs Actor-Critic.
2.  **Convergencia**: Velocidad de aprendizaje en episodios.
3.  **Desempeño Final**: Recompensa media en episodios de test (modo determinista).

Típicamente, **Actor-Critic** (especialmente con Dueling/Double DQN) debería mostrar una convergencia más estable y rápida que REINFORCE en entornos complejos como LunarLander, aunque REINFORCE puede ser muy competitivo en entornos simples como CartPole.


