🧠 Chat LLM Local con OpenVINO + Carga de PDFs + Búsqueda en Internet
Versión Mayo 2025 — Autor: JC Llanos V.

📧 jcllanosv@hotmail.com

🔗 https://www.linkedin.com/in/jcllanosv/

🚀 Descripción General

Este proyecto implementa un chat conversacional local potenciado por modelos LLM optimizados con OpenVINO, permitiendo:

✔ Elección de múltiples modelos (TinyLlama, Phi, Mistral, DeepSeek, Qwen, Gemma, StarCoder2, etc.)
✔ Ejecución acelerada en CPU o Intel iGPU
✔ Guardado automático de modelos convertidos para uso offline
✔ Lectura y carga de archivos PDF con extracción automática de texto
✔ Comandos especiales para resumir, comentar y manejar contenido del PDF
✔ Integración opcional de búsquedas en Internet (DuckDuckGo Search) para respuestas con contexto actualizado
✔ Historial conversacional acotado para mantener el rendimiento
✔ Interfaz simple por consola y selector de PDF mediante una ventana GUI con Tkinter

Este programa funciona completamente local, salvo cuando se activa la opción de búsqueda en internet.

📦 Funcionalidades Principales
🔍 1. Selección de Modelos LLM

El script incluye una lista extensa de modelos ya optimizados para OpenVINO, incluyendo:

TinyLlama

Phi-2 y Phi-1.5

StarCoder2 (enfocado en código)

CodeGen2

DeepSeek-R1 Qwen (1.5B y 7B)

Mistral 7B Instruct

Zephyr 7B

Dolly v2

RedPajama

Gemma 2B

Qwen2 (0.5B y 1.5B)

DistilGPT-2

TinyLlama original

Muchos vienen en formato INT4/INT8 para máximo rendimiento en GPU Intel.

Todos los modelos se descargan o convierten una sola vez y luego se reutilizan desde la carpeta:

mis_modelos_openvino/

📄 2. Carga y Procesamiento de PDF

Incluye soporte completo para PDFs mediante PyMuPDF:

Selector gráfico para elegir archivos (tkinter)

Extracción completa de texto

Advertencias si el PDF no contiene texto

Límite configurable de caracteres para evitar prompts excesivos

Comandos disponibles:

Comando	Función
!cargar_pdf	Selecciona un PDF desde el explorador
!resumir_pdf	Genera un resumen usando el LLM
!comentar_pdf	Analiza y comenta el PDF
!olvidar_pdf	Borra el PDF cargado del contexto
🌐 3. Búsqueda en Internet (opcional)

Al activar:

ENABLE_INTERNET_SEARCH = True


El asistente puede:

Buscar información reciente mediante DuckDuckGo

Resumir resultados

Incluir contexto web en las respuestas

Ideal para respuestas que requieren actualidad (ej.: leyes, noticias, eventos recientes).

🧩 4. Chat Interactivo y Persistente

El sistema mantiene:

Historial de turnos configurable

Manejo robusto de errores

Control automático de temperatura, top_k, top_p y generación de tokens

Limpieza automática del contexto en cada carga de modelo

🛠️ Tecnologías Utilizadas
Tecnología	Uso
Python 3.10+	Lenguaje principal
OpenVINO Runtime	Inferencia acelerada
Optimum Intel (HuggingFace)	Carga/convertir modelos LLM
Transformers	Tokenización
PyMuPDF (fitz)	Lectura de PDF
Tkinter	Selector gráfico de archivos
DuckDuckGo Search (DDGS)	Búsquedas web (opcional)
📁 Estructura del Proyecto
/
├── mis_modelos_openvino/     # (Se crea automáticamente)
├── README.md                 
└── main.py                   # Este script principal

▶️ Cómo Ejecutarlo
1. Instalar dependencias
pip install openvino==2024.4.0
pip install optimum[intel]
pip install transformers
pip install PyMuPDF
pip install duckduckgo_search
pip install tkinter  # En Linux puede requerir paquete del sistema

2. Ejecutar el script
python main.py

3. Seleccionar un modelo

El programa mostrará un listado, por ejemplo:

 1. OpenVINO: TinyLlama 1.1B Chat (INT8)
 2. OpenVINO: Mistral 7B Instruct (INT4)
 3. OpenVINO: DeepSeek-Qwen 1.5B (INT4)
 ...

🧭 Cómo Usarlo

Una vez cargado el modelo:

Listo para chatear...
Comandos PDF: !cargar_pdf, !resumir_pdf, !comentar_pdf, !olvidar_pdf


Ejemplo básico:

Tú: ¿Cuál es la capital de Chile?


Ejemplo usando PDF:

Tú: !cargar_pdf
Tú: !resumir_pdf
Tú: Explica el capítulo 2 del PDF


Ejemplo con internet:

Tú: ¿Cuál es la situación actual de la Ley 21.659 en Chile?

⚠️ Consideraciones

Algunos modelos requieren trust_remote_code=True.

Modelos grandes (7B+) pueden requerir GPU Intel Arc o mucha RAM.

La búsqueda en internet puede producir resultados variables según región (cl-es preconfigurado).

🧑‍💻 Autor

JC Llanos V.
🔗 https://www.linkedin.com/in/jcllanosv/
