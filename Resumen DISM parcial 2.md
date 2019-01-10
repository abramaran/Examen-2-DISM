# Resumen DISM parcial 2

## 2. Cálculo de rendimientos

Métrica más utilizada para medir la capacidad de cálculo -> **FLOPS** o **Floating-Point Operations Per Second**.

### 2.1 Rendimiento Teórico y Ley de Amdahl

Cuantificar el rendimiento teórico que podemos obtener al acelerar una aplicación mediante CUDA. Saber si merece la pena.

Para cuantificar este rendimiento se utiliza la ley de Amdahl.

El **factor máximo de mejora** depende de:

- **% de tiempo** que consuma la porción acelerable.
- **Factor de aceleración** que podamos obtener.

![1547071720450](.\Resumen DISM parcial 2.assets\1547071720450.png "Factor máximo de mejora")

Al final suele ser más conveniente acelerar la parte que más tiempo ocupe si la diferencia es importante.

El **factor de aceleración** depende de:

- Si el problema es fácilmente paralelizable.
- De la adecuación del código para CPU y GPU.
- Si el flujo de datos y de control se adaptan al modelo de cómputo de la GPU.
- Optimización del acceso a memoria.

Por otro lado, la aceleración global dependerá del peso de la fracción de código acelerada en el tiempo de ejecución global.

Ecuación resultante:
$$
S = \frac{1}{(1-F_m) + \frac{F_m}{A_m}}
$$
Siendo

- ***Fm***: fracción de tiempo que el sistema utiliza el subsistema mejorado
- ***Am***: factor de mejora que se ha introducido en el subsistema mejorado.

### 2.7 Trivia sobre el rendimiento

1. El **consumo energético** y el **calor** son los principales obstáculos a la hora de meter más transistores en un mismo chip, junto al **efecto túnel** de las puertas lógicas de un tamaño inferior a 7 nanómetros.
2. El paralelismo a nivel de instrucción ocupa mucha circuitería y su rendimiento está saturado.
3. Las **operaciones aritméticas** son **rápidas** y los accesos a **memoria** muy **lentos**.
4. La **ley de Moore ha dejado de cumplirse**: nos hemos topado con las barreras físicas que impiden seguir comprimiendo los transistores.

## 3. CUDA

Ecosistema enfocado a la **programación masivamente paralela.**

### 3.1 Arquitectura hardware

Modelo de procesamiento: **Single Instruction Multiple Thread (SIMT)**

#### 3.1.1 Situación física e interconexión

- **GPU**: Tarjeta de expansión. Se conecta al sistema mediante puertos **PCIe** presentes en la placa base.
- **Chipset**: Gestionar conexiones de la CPU con el resto de componentes.
  - Southbridge: Gestiona periféricos (ratón, teclado, etc)
  - Northbridge: Gestiona buses de gráficos, comunicación con la memoria de la CPU![1547143292204](.\Resumen DISM parcial 2.assets\1547143513365.png)
- CPUs integradas: La tarjeta gráfica se encuentra integrada en 

