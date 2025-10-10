# Calculador de Similitud Semántica

Un servidor FastAPI que compara la similitud semántica entre dos cadenas de texto utilizando un modelo de sentence-transformers.

## Características

- Comparación de similitud semántica entre cadenas de texto
- Utiliza modelo de sentence-transformers (all-MiniLM-L6-v2 por defecto)
- Utiliza automáticamente GPU si está disponible
- API REST simple con solicitud/respuesta JSON
- Almacenamiento en caché del modelo para evitar descargas repetidas al reiniciar el servidor

## Instalación

1. Clona este repositorio:
```bash
git clone https://github.com/yourusername/similarity-scorer.git
cd similarity-scorer
```

2. Instala los paquetes requeridos:
```bash
pip install -r requirements.txt
```

## Uso

### Opción 1: Ejecutar directamente con Python

#### Modo Desarrollo
```bash
uvicorn main:app --reload
```

#### Modo Producción con Gunicorn
```bash
./start.sh
```

Puedes personalizar la configuración estableciendo variables de entorno:
```bash
MODEL_NAME=all-mpnet-base-v2 WORKERS=4 ./start.sh
```

El servidor se ejecutará en: `http://127.0.0.1:16000`

La documentación interactiva de la API está disponible en: `http://127.0.0.1:16000/docs`

### Opción 2: Ejecutar con Docker

```bash
docker build -t similarity-scorer .
docker run -p 16000:16000 similarity-scorer
```

### Opción 3: Ejecutar con Docker Compose

```bash
docker-compose up
```

Para ejecutar en modo separado:
```bash
docker-compose up -d
```

Para detener el servicio:
```bash
docker-compose down
```

## Gestión del Servidor

### Iniciando el Servidor

Para iniciar el servidor en modo producción:

```bash
./start.sh
```

Para entornos con memoria limitada:

```bash
./start_optimized.sh
```

### Deteniendo el Servidor

Para detener cualquier instancia del servidor en ejecución:

```bash
./terminate.sh
```

Si la terminación regular no funciona, utiliza el script de terminación forzada (puede requerir sudo):

```bash
sudo ./force_terminate.sh
```

Estos scripts harán:

1. Identificar todos los procesos en ejecución relacionados con el calculador de similitud
2. Intentar terminarlos de manera elegante (script regular) o forzada (script de fuerza)
3. Reportar el estado después del intento de terminación

## Endpoints de la API

### POST /compare

Compara la similitud semántica entre dos oraciones.

**Cuerpo de la Solicitud:**
```json
{
  "sentence1": "¿Cómo horneo un pastel?",
  "sentence2": "¿Cuál es el proceso para hacer un pastel?"
}
```

**Respuesta:**
```json
{
  "sentence1": "¿Cómo horneo un pastel?",
  "sentence2": "¿Cuál es el proceso para hacer un pastel?",
  "semantic_similarity": 0.87
}
```

## Personalización

### Cambiando el Modelo

Puedes cambiar el modelo de dos maneras:

1. Estableciendo la variable de entorno `MODEL_NAME`:

   ```bash
   # Al ejecutar con Python
   MODEL_NAME=all-mpnet-base-v2 uvicorn main:app --reload

   # Al ejecutar con Docker
   docker run -p 16000:16000 -e MODEL_NAME=all-mpnet-base-v2 similarity-scorer

   # O actualiza la variable de entorno en docker-compose.yml
   # y luego ejecuta docker-compose up
   ```

2. Editando directamente el valor por defecto en `main.py`:

   ```python
   model_name = os.environ.get("MODEL_NAME", "all-mpnet-base-v2")
   ```

Nota que modelos más precisos como "all-mpnet-base-v2" requieren más recursos computacionales pero proporcionan mejores resultados de similitud.

## Consideraciones de Rendimiento

### Almacenamiento en Caché del Modelo

La aplicación está configurada para almacenar en caché los modelos descargados en el directorio `models/`. Esto significa:

1. El modelo se descargará solo una vez, incluso si reinicias el servidor múltiples veces
2. Los inicios posteriores del servidor serán mucho más rápidos
3. Al usar Docker, el caché del modelo se almacena en un volumen nombrado para persistencia

Esto es especialmente importante para modelos más grandes como "all-mpnet-base-v2" que pueden ser de varios cientos de MB de tamaño.

### Proceso de Modelo Compartido

Para optimizar el uso de memoria, la aplicación utiliza un servicio de modelo dedicado que se ejecuta en un proceso separado:

1. Solo se carga una copia del modelo en memoria independientemente de cuántos trabajadores Gunicorn estén ejecutándose
2. Todos los procesos trabajadores se comunican con el servicio de modelo vía IPC (comunicación interprocesos)
3. Esto reduce significativamente el uso de memoria al ejecutar con múltiples trabajadores

Esta arquitectura es particularmente beneficiosa para modelos grandes que de otra manera consumirían varios GB de RAM si se cargaran por separado en cada proceso trabajador.

## Optimización de Memoria

Esta aplicación está diseñada para ser eficiente en memoria, especialmente en entornos con recursos limitados:

1. **Instancia Única del Modelo**: El modelo se carga solo una vez en un proceso dedicado y se comparte entre todos los trabajadores
2. **Conteo Conservador de Trabajadores**: La configuración de Gunicorn utiliza menos trabajadores que lo típico para reducir el uso de memoria
3. **Monitoreo de Memoria**: La aplicación registra el uso de memoria en varios puntos para rastrear la utilización de recursos
4. **Conteo Configurable de Trabajadores**: Puedes establecer la variable de entorno `WORKERS` para limitar aún más los trabajadores
5. **Gestión del Ciclo de Vida de Trabajadores**: Los trabajadores se reinician después de manejar un cierto número de solicitudes para prevenir fugas de memoria

### Inicio Optimizado

Para entornos con memoria muy limitada, utiliza el script de inicio optimizado:

```bash
./start_optimized.sh
```

Este script establece variables de entorno para optimizar el uso de memoria y proporciona seguimiento del uso de memoria.

### Soporte de GPU

La aplicación detecta automáticamente y utiliza recursos GPU disponibles:

#### Soporte de GPU en macOS (Apple Silicon/AMD)

En macOS, la aplicación utiliza:
- Metal Performance Shaders de Apple (MPS) en Macs con Apple Silicon (M1/M2/M3)
- GPUs AMD en Macs Intel con tarjetas gráficas compatibles

Para verificar si tu Mac está utilizando aceleración GPU:
1. Inicia el servidor
2. Visita `http://127.0.0.1:16000/system-info` para ver la información del dispositivo
3. Busca `"mps_available": true` y `"current_device": "mps"` en la respuesta

#### Soporte de GPU NVIDIA (Linux/Windows)

En sistemas con GPUs NVIDIA, CUDA se utilizará automáticamente.

Si ejecutas en Docker, asegúrate de incluir la configuración de runtime de GPU como se especifica en el archivo docker-compose.yml.

## Licencia

[Licencia MIT](LICENSE)