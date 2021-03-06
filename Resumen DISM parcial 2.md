# Resumen DISM parcial 2

*Alberto Benavent Ramón, 2019*

## 2. Cálculo de rendimientos

Métrica más utilizada para medir la capacidad de cálculo -> **FLOPS** o **Floating-Point Operations Per Second**.

### 2.1 Rendimiento Teórico y Ley de Amdahl

Cuantificar el rendimiento teórico que podemos obtener al acelerar una aplicación mediante CUDA. Saber si merece la pena.

Para cuantificar este rendimiento se utiliza la ley de Amdahl.

El **factor máximo de mejora** depende de:

- **% de tiempo** que consuma la porción acelerable.
- **Factor de aceleración** que podamos obtener.

![1547071720450](Resumen DISM parcial 2.assets\1547071720450.png "Factor máximo de mejora")

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
  - Northbridge: Gestiona buses de gráficos, comunicación con la memoria de la CPU![1547143292204](Resumen DISM parcial 2.assets\1547143513365.png)
- CPUs integradas: La tarjeta gráfica se encuentra integrada en el *Northbridge*. Comparte tanto controlador de memoria como memoria física con la CPU.![1547145414600](Resumen DISM parcial 2.assets/1547145414600.png)
- Otras configuraciones:
  - Múltiples GPUs varios puertos PCIe.
  - Múltiples GPUs intercomunicadas: Scalable Link Interface (SLI) (NVIDIA), CrossFire (AMD). Ventaja: Menor latencia y mayor ancho de banda al no pasar por el Northbridge.![1547145665054](Resumen DISM parcial 2.assets/1547145665054.png)

#### 3.1.2 Single Instruction Multiple Threads (SIMT)

##### Taxonomía de Flynn

![1547145795947](Resumen DISM parcial 2.assets/1547145795947.png)

**GPUs**: Combinan **SIMD** con procesadores **multihilo**: **Single Instruction Multiple
Thread (SIMT)**.

**SIMT: Un procesador** ejecuta **múltiples hilos** de manera **concurrente** sobre **diferentes datos**.

Arquitectura de una CPU:

- SISD: procesadores **monolíticos** completamente **secuenciales.**
- SIMD: **Multinúcleo** o extensiones vectoriales.

##### Diferencias de arquitectura entre CPU y GPU

| CPU                                                 | GPU                                                          |
| --------------------------------------------------- | ------------------------------------------------------------ |
| Cachés grandes                                      | Cachés reducidos                                             |
| Unidad de control compleja                          | Unidad de control simple                                     |
| ALUs complejas y poco numerosas con escasa latencia | Gran número de ALUs simples centradas en la cantidad de trabajo |
| Poca latencia, poca cantidad de trabajo             | Mucha latencia, mucha cantidad de trabajo                    |

![1547146447615](Resumen DISM parcial 2.assets/1547146447615.png "Ilustración gráfica de las diferencias entre los recursos hardware de una CPU y una GPU.")

#### 3.1.3 Streaming Multiprocessors (SMs)

Una GPU CUDA se descompone en:

- **Interfaz** que conecta GPU al bus PCIe. Se encarga también de la sincronización GPU-CPU.
- **Copy engines**. Transferencias de memoria asíncronas. Entre 0 y 2. 
- Interfaz de memoria **Dynamic RAM (DRAM)**. **Jerarquía de cachés** que conecta la GPU a la memoria principal. *Esto lo vimos cuando la segunda práctica de CUDA.*
- **Texture Processing Clusters (TPCs)** o **Graphics Processing Clusters (GPCs)**. Contienen los SMs y caché.

**Streaming Multiprocessors (SMs)**: unidades de cómputo de la arquitectura CUDA. Se agrupan en GPCs.

Cada SM está formado por:

- **Núcleos CUDA**. Unidades de ejecución. Operaciones aritméticas enteras o float.
- Unidades de **funciones especiales**. Aproximaciones de operaciones complejas.
- **Warp schedulers**. Distribuir warps a los núcleos.
- Caché de constantes.
- Memoria compartida para todos los hilos.
- Archivo de registros.

![1547147924292](Resumen DISM parcial 2.assets/1547147924292.png)

### 3.2 Arquitectura software

CUDA está compuesto por una serie de capas (stack):

![1547148622079](Resumen DISM parcial 2.assets/1547148622079.png)

#### 3.2.2 Compilación

El compilador decide lo que va a la **CPU** (**Host**) y a la **GPU** (**Device**). Lo de la GPU se pasa a un **lenguaje intermedio**, **Parallel Thread eXecution (PTX)** . Esto sirve para que se pueda ejecutar en muchas tarjetas gráficas distintas, lo interpreta el driver.

![1547149260260](Resumen DISM parcial 2.assets/1547149260260.png)

#### 3.2.3 Conceptos básicos

##### Dispositivo

Cada GPU física del sistema.

##### Contexto

Equivalente a los procesos de la CPU pero en la GPU. Contiene los objetos que forman un programa CUDA.

##### Kernels

Funciones cuya ejecución:

- Es ordenada de forma asíncrona por la CPU.
- Se realiza en la GPU de forma masivamente paralela.

Tres tipos de funciones:

|              | `__host__` | `__device__` | `__global__` |
| ------------ | :--------: | :--: | :--: |
| Lanzada en   | CPU | GPU | CPU |
| Se ejecuta en | CPU | GPU | GPU |

Invocación de kernel:

```c++
// Con el API Runtime
kernel <<< malla, bloque, memoria_compartida, stream >>> (parámetro1, parámetro2, ..., parámetroN); // memoria_compartida y stream son opcionales y se pueden omitir.

// Con la API del driver
CUResult cuLaunchKernel (
    CUfunction kernel,
    unsigned int gridDimX,
    unsigned int gridDimY,
    unsigned int gridDimZ,
    unsigned int blockDimX,
    unsigned int blockDimY,
    unsigned int blockDimZ,
    unsigned int shmem,
    CUstream stream,
    void ** params,
    void ** extra);
```

Invo

Definir tamaños de malla y bloque:

```c++
dim3 block_dim (int 1d, int 2d, int 3d);
dim3 grid_dim (int 1d, int 2d, int 3d);
```

Si no queremos usar una dimensión le pasamos un 1.

##### Hilos

Pues eso, hilos de ejecución los clásicos *threads*. Cada hilo ejecuta una copia del kernel. Dispone de:

- Variables pasadas por parámetro (punteros memoria GPU)
- Constantes globales en memoria GPU
- Variables especiales de identificación:
  - gridDim: tamaño de la malla
  - blockDim: tamaño de los bloques
  - blockIdx: identificador de bloque
  - threadIdx: identificador de hilo

Direccionamiento: 
$$
idx = blockId.x · blockDim + threadId
$$


##### Warps

Unidad que agrupa 32 hilos.

##### Bloques

Conjunto de hilos. Puede tener 1, 2 o 3 dimensiones. Cada bloque se ejecuta sobre un único SM de forma íntegra. Un SM puede tener varios bloques.

Es común elegir tamaños de bloque múltiplo de un warp (32 hilos). Esto se supone que tiene una ligera mejora en el rendimiento.

##### Mallas

Conjunto de bloques. Puede tener 1, 2 o 3 dimensiones.

![1547230832653](Resumen DISM parcial 2.assets/1547230832653.png)

#### 3.2.4 Flujo de ejecución

1. Inicialización GPU
2. Reserva de memoria en host y device
3. Copia de datos de host a device
4. Lanzamiento de kernel
5. Copia de datos de device a host
6. Repetir del 3. al 5. como sea necesario
7. Liberación de memoria, fin del proceso

![1547209381892](Resumen DISM parcial 2.assets/1547209381892.png)

#### 3.2.5 Evolución generacional

La capacidad de cómputo de cada generación de tarjetas gráficas CUDA se conoce como Compute Capability. Para saber las características de la nuestra, lo mejor es ejecutar el ejemplo en *CUDA Samples/1_Utilities/deviceQuery*.

## 5. Modelo de procesamiento

### 5.3 Limitaciones de memoria

- Recursos reservados al inicio
- Registros limitados por hilo
- Memoria local limitada por hilo
- La memoria global no es infinita
- No hay memoria virtual ni swapping

Cada hilo necesita memoria para ejecutarse y para los datos. Suelen utilizar una abstracción de la memoria de la GPU, la memoria local. No hay paginación virtual en la GPU. Reserva anticipada de la memoria por hilo.

### 5.4 Limitaciones de tiempo

Situaciones donde la GPU se usa para gráficos y CUDA al mismo tiempo: el sistema operativo y el driver fijan un **tiempo máximo** de ejecución, transcurrido el cual la gráfica cambia otra vez a sus funciones de visualización.

### 5.5 Escalabilidad transparente y planificación

**No todos los bloques son ejecutados de manera concurrente**

**No hay ninguna garantía respecto a su orden de ejecución más allá de que se ejecutarán en el mismo SM**

![1547295391361](Resumen DISM parcial 2.assets/1547295391361.png)

En la imagen se ve que cuando ejecutas el programa CUDA tú no sabes en qué SM ni en qué orden se van a ejecutar los bloques, y también que no se ejecutan todos simultáneamente.

Una vez un bloque ha sido asignado a un SM, sus hilos son ejecutados siguiendo un modelo SIMD, en agrupaciones de 32 hilos consecutivos denominadas warps. Los **hilos** (o *lanes*) **de un warp** se ejecutan **físicamente en paralelo**.

Todos los hilos de un warp ejecutan la misma instrucción, y son elegidos según una cola de prioridad para ejecución.

### 5.6 Sincronización y control de flujo

Para sincronizar **hilos de un mismo bloque** (que sabemos que están en el mismo SM):

- **Barreras de sincronización**. `__syncthreads()` detiene la ejecución hasta que todos los hilos del bloque han llegado a ese punto.
- **Operaciones atómicas**. Lectura, escritura y modificación en memoria global o compartida. Estas funciones aseguran que solamente un hilo cada vez trabaje con la misma posición de memoria.
- **Memoria compartida**. No dice cómo pero se supone que puede realizar algunos tipos de sincronización.

### 5.7 Control de flujo

Se pueden utilizar condicionales en los kernels CUDA.

```c++
if (x < 0.0)
	z = x - 2.0;
else
	z = sqrt(x);
```

¡Pero **todos los hilos de un warp deben ejecutar la misma instrucción**! Así que el compilador **realiza las condiciones antes de ejecutar**, se ejecutarán todos los hilos que entren en el if, y luego el resto. Los hilos que no se ejecuten estarán marcados con un flag equivalente al NOP (no operation creo).

La misma condición en predicado:

```c++
cond: p = (x < 0.0);
p: z = x - 2.0;
!p: z = sqrt(x);
```
Divergencia (Granularidad inferior al tamaño del warp)
```c++
if (threadIdx.x > 2) 
  dosomething;
else
  dootherthing;
```

No divergencia (Granularidad múltiplo del tamaño del warp)

```c++
if (threadIdx.x / WARP_SIZE > 2) 
  dosomething;
else
  dootherthing; 
```

### 5.8 Streams

Un kernel se lanza de manera asíncrona, es decir, el programa del host puede seguir ejecutándose nada más lanzarlo. Para solapar cómputo y otras operaciones bloqueantes se usan **streams**. Además de la concurrencia de grano fino de los hilos, tenemos concurrencia de grano grueso de la siguiente forma:

- Entre **CPU y GPU** por lo de que lanzar un kernel es asíncrono.
- Entre **cómputo de kernel y transferencia de memoria**. Las copy engines de la GPU actúan independientemente de los SMs.
- Entre **kernels**. Con SMs de generación 2.X+ pueden ejecutar hasta 4 kernels en paralelo.
- **Multi-GPUs**.

#### ¿Qué son?

Abstracciones: Secuencia de operaciones que se ejecutan de forma secuencial según el orden
de envío en la GPU.

- Operaciones de varios streams: Pueden ejecutarse de manera concurrente según recursos disponibles.
- Operaciones de varios streams y distinta naturaleza: Pueden solaparse.

### 5.9 Medición de tiempos, sincronización host-device y eventos.

La medición de tiempos se suele ejecutar en el host, o bien con herramientas del sistema o con cronómetros de CUDA.

Las **transferencias de memoria** entre device y host son **bloqueantes**.

```c++
cudaMemcpy(d_x_, h_x_, vector_size_bytes_, cudaMemcpyHostToDevice);
// Host bloqueado hasta finalizar copia
cudaMemcpy(d_y_, h_y_, vector_size_bytes_, cudaMemcpyHostToDevice);
// Host bloqueado hasta finalizer copia

kernel<<<grid, block>>>(d_x_, d_y_, N);
// Host desbloqueado, continua ejecución nada más emitir la orden!

cudaMemcpy(h_y_, d_y_, vector_size_bytes_, cudaMemcpyDeviceToHost);
// Host bloqueado hasta finalizer copia
```

Para medir el tiempo de ejecución del kernel de manera sencilla podemos bloquear el host hasta que acabe la ejecución del kernel.

```c++
cudaMemcpy(d_x_, h_x_, vector_size_bytes_, cudaMemcpyHostToDevice);
// Host bloqueado hasta finalizar copia
cudaMemcpy(d_y_, h_y_, vector_size_bytes_, cudaMemcpyHostToDevice);
// Host bloqueado hasta finalizer copia

unsigned long t_start_ = cpu_timer();
kernel<<<grid, block>>>(d_x_, d_y_, N);
// Host desbloqueado, continua ejecución nada más emitir la orden!
cudaDeviceSynchronize();
// Host bloqueado hasta que la GPU termine el último comando
unsigned long t_end_ = cpu_timer();

Unsigned long elapsed_ = t_end_ - t_start_;

cudaMemcpy(h_y_, d_y_, vector_size_bytes_, cudaMemcpyDeviceToHost);
// Host bloqueado hasta finalizar copia
```

Lo importante en ese código es el `cudaDeviceSynchronize()` que espera a que acabe la ejecución del kernel.

Para no bloquear la CPU y medir el tiempo de ejecución tendríamos que usar **eventos**.

```c++
cudaEvent_t t_start_, t_stop_;
cudaEventCreate(&t_start_);
cudaEventCreate(&t_stop_);

cudaMemcpy(d_x_, h_x_, vector_size_bytes_, cudaMemcpyHostToDevice);
cudaMemcpy(d_y_, h_y_, vector_size_bytes_, cudaMemcpyHostToDevice);

cudaEventRecord(t_start_);
kernel<<<grid, block>>>(d_x_, d_y_, N);
cudaEventRecord(t_stop_);

cudaMemcpy(h_y_, d_y_, vector_size_bytes_, cudaMemcpyDeviceToHost);
cudaEventSynchronize(t_stop_);
float milliseconds_ = 0.0f;
cudaEventElapsedTime(&milliseconds_, t_start_, t_stop_);
```



## 7 Memorias CUDA

### 7.1 Jerarquía de memorias

- **Localidad espacial**. Cercanía física en la memoria del sistema de dos posiciones de memoria. En plan, si de verdad están pegadas.
- **Localidad temporal**. Cercanía temporal en el acceso a varias posiciones de memoria. Ambas cosas suelen ir de la mano.

Como una memoria rápida es cara, se utiliza una jerarquía de cachés, sucesivamente más rápidos y más pequeños, a través de los cuales se copian bloques de uno a otro. Se asume que las posiciones de memoria tendrán una baja localidad temporal y espacial.

- CPU:

![1547313272218](Resumen DISM parcial 2.assets/1547313272218.png)

- GPU:

![1547313306426](Resumen DISM parcial 2.assets/1547313306426.png)

### 7.2 Tipos de memoria en la GPU

![1547313380917](Resumen DISM parcial 2.assets/1547313380917.png)

#### 7.2.1 Global

- **Compartida** por todos los SMs
- **Alta latencia** y **no cacheable**
- Alberga **datos de la aplicación**
- Reserva desde la CPU con `cudaMalloc()`
- Liberación desde CPU con `cudaFree()`
- Transferencia CPU-GPU con `cudaMemcpy()`

#### 7.2.2 Local

- **Información** local **de cada hilo**
- Parte de DRAM (global) pero **cacheable** (L1 y L2)
- Alberga **variables y vectores que no caben en registros**
- Gestión **automática** desde GPU

#### 7.2.3 Memoria de constantes

- **Compartida** por todos los SMs
- **Optimizada** para **solo lectura**
- Alberga datos **constantes** (pero **pueden ser modificados**)
- Reserva estática desde CPU (`__constant__`)
- Transferencia CPU-GPU (`cudaMempcyToSymbol()`, `cudaMemcpyFromSymbol()`)

#### 7.2.4 Memoria de texturas

- **Compartida** por todos los SMs

- **Cacheable** para **solo lectura**

- **Optimizada** para **no coalescente**

- Alberga datos **constantes**

- Interpolación o normalización

- Declarada con tipo de datos (`cudaArray`)

#### 7.2.5 Registros

- Cada SM posee una **cantidad fija**

- Repartidos entre los hilos

- Extremadamente **baja latencia**

- Albergan **variables de los kernels**

- Relativamente **limitados** en cantidad (depende del número de hilos en ejecución)

#### 7.2.6 Memoria compartida

- Cada SM posee una **cantidad fija**

- **Compartida** por los hilos de **un mismo bloque**

- **Latencia baja**

- Albergan **datos reutilizables**

- Reserva en la GPU (`__shared__`)

- Copia de datos manual a posiciones de memoria en kernel

### 7.4 Consideraciones de rendimiento en el uso de memorias GPUs

La memoria no es ilimitada, *obviamente*.

#### 7.4.1 Register spilling

Si la aplicación necesita más registros por hilo de los disponibles:
1. Desbordan a memoria local.
2. Ciertas variables se convierten en arrays para almacenar un elemento por hilo.

Esto conlleva una **pérdida de rendimiento**.

#### 7.4.2 Acceso coalescente

El acceso a posiciones de memorias coalescentes, es decir, próximas en la memoria física, mejora el rendimiento ya que aprovecha la jerarquía de memorias.![1547314629490](Resumen DISM parcial 2.assets/1547314629490.png)

#### 7.3.9 Memoria compartida

Variables que van a ser compartidas por todos los hilos de un bloque. 

- Forma de comunicación o sincronización entre hilos de un bloque.
- Mejora la reutilización de datos y reduce los accesos a memoria global.
- Reduce potencialmente el número de registros necesarios por hilo.

Declaración:

```c++
__global__ void medianFilter2D_sm(unsigned char *d_output, unsigned char *d_input)
{
  // Declaración de memoria compartida para almacenar una porción de una imagen
   __shared__ unsigned char d_input_sm_[tamaño de porción compartida];
	[…]
}
```

Copia de datos de global a compartida:

```c++
__global__ void medianFilter2D_sm(unsigned char *d_output, unsigned char *d_input)
{
	[…]
  // Copia de datos manual de global a compartida
   d_input_sm[idx_compartida_] = d_input[idx_global_];
	[…]
}
```

Sincronización

```c++
__global__ void medianFilter2D_sm(unsigned char *d_output, unsigned char *d_input)
{
	[…]
  // Sincronización de hilos para asegurar que todos los datos han sido copiados
   __syncthreads();
	[…]
}
```

Utilización de datos compartidos:

```c++
__global__ void medianFilter2D_sm(unsigned char *d_output, unsigned char *d_input)
{
	[…]
  // Utilización de datos de memoria compartida
	north_ = d_input_sm_[idx_north_sm_];
	north_east_ = d_input_sm_[idx_north_east_sm_];
	[…]
}
```

