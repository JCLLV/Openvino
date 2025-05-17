# Ejemplo de aplicación con uso de GPU Intel de Laptop con Openvino
Repositorio con programa de ejemplo de análisis de archivo PDF utilizando la GPU Intel de un laptop.

Aquí esta el código completo del script.

Selección de Modelo: Elige de una lista al inicio.
Guardado/Carga Local: Los modelos se guardan en mis_modelos_openvino/ después de la primera descarga/conversión y se cargan desde allí en usos posteriores.
Barras de Progreso: tqdm (si está instalado) debería mostrar progreso durante las descargas de Hugging Face.
Chat Interactivo con Historial: Mantiene un contexto de los últimos turnos de la conversación.
Búsqueda en Internet (RAG): Usa duckduckgo-search para obtener contexto actualizado.
Carga de PDF con Diálogo Gráfico: Usa tkinter para que puedas seleccionar un PDF gráficamente.
Interacción con PDF: Comandos para cargar, resumir, comentar y olvidar PDFs.
Corrección del OverflowError: Manejo más robusto de max_length para el tokenizador.
Parámetros de Generación Ajustables.
Antes de ejecutar:

Asegúrate de tener todas las librerías instaladas en tu entorno openvino_env_py311:
pip install openvino openvino-dev torch transformers accelerate "optimum[openvino]" duckduckgo-search PyMuPDF tqdm
Importante: Si tienes una carpeta mis_modelos_openvino/ de ejecuciones anteriores, considera borrarla (o las subcarpetas de los modelos que quieras probar de nuevo) antes de ejecutar este script por primera vez. Esto asegurará que la lógica de guardado/carga local se pruebe desde cero.

