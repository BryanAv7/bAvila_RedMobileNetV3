// ----------------------------------------------------------------------------------------------------------
// Modelo MobileNetV3 para detección de objetos 
// Nombre: Bryan Avila
// Carrera: Computación
// Materia: Visión por Computadora
// Fecha: 2025-06-28
// ----------------------------------------------------------------------------------------------------------

// Librerías por utilizar
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <onnxruntime_cxx_api.h>

// Espacio reservado de nombres
using namespace std;
using namespace cv;
using namespace std::chrono;

// Crear entorno ONNX Runtime
Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "MobileNetV3");


// ----------------------------------------------------------------------------------------------------------
// Función: load_labels
// Descripción: Carga las etiquetas de las clases
// ----------------------------------------------------------------------------------------------------------
vector<string> load_labels(const string& filename) {
    vector<string> labels;
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error al abrir archivo de etiquetas: " << filename << endl;
        return labels;
    }
    string line;
    while (getline(file, line)) {
        labels.push_back(line);
    }
    return labels;
}


// ----------------------------------------------------------------------------------------------------------
// Función: preprocess
// Descripción: Preprocesamiento de la imagen para el modelo
// ----------------------------------------------------------------------------------------------------------
Mat preprocess(const Mat& frame, int input_width, int input_height) {
    // Redimensionar en CPU
    Mat resized;
    resize(frame, resized, Size(input_width, input_height));

    // Subir imagen a la GPU
    cv::cuda::GpuMat gpu_resized;
    gpu_resized.upload(resized);

    // Convertir BGR a RGB en GPU
    cv::cuda::GpuMat gpu_rgb;
    cv::cuda::cvtColor(gpu_resized, gpu_rgb, COLOR_BGR2RGB);

    // Descargar imagen procesada de vuelta a la CPU
    Mat output;
    gpu_rgb.download(output);

    return output;
}


// ----------------------------------------------------------------------------------------------------------
// Función principal
// ----------------------------------------------------------------------------------------------------------
int main() {
    // Rutas de archivos del modelo, video y etiquetas
    string model_path = "/home/bryan/Documentos/segundoInterciclo/TrabajoU4PartB/models/mobileNetV3.onnx";
    string video_path = "/home/bryan/Documentos/segundoInterciclo/TrabajoU4PartB/video4.mp4";
    string labels_path = "/home/bryan/Documentos/segundoInterciclo/TrabajoU4PartB/models/coco.names";

    // Cargar etiquetas desde archivo
    vector<string> class_names = load_labels(labels_path);
    if (class_names.empty()) {
        cerr << "No se cargaron las etiquetas del modelo" << endl;
        return -1;
    }

    // Configurar ejecución del modelo (por defecto CPU)
    bool usarGPU = false; // False: CPU / True: GPU
    Ort::SessionOptions session_options;

    // Configuración para ejecutar en GPU (si se activa)
    // if (usarGPU) {
    //     if (OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0) != nullptr) {
    //         cerr << "Error al habilitar CUDA" << endl;
    //         return -1;
    //     }
    //     cout << "Usando GPU" << endl;
    // }

    cout << "Usando CPU" << endl;

    // Crear sesión ONNX
    Ort::Session session(env, model_path.c_str(), session_options);
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Obtener nombres de entradas y salidas del modelo
    vector<const char*> input_names;
    vector<const char*> output_names;
    vector<Ort::AllocatedStringPtr> input_name_ptrs;
    vector<Ort::AllocatedStringPtr> output_name_ptrs;

    size_t num_inputs = session.GetInputCount();
    for (size_t i = 0; i < num_inputs; ++i) {
        input_name_ptrs.emplace_back(session.GetInputNameAllocated(i, Ort::AllocatorWithDefaultOptions()));
        input_names.push_back(input_name_ptrs.back().get());
    }

    size_t num_outputs = session.GetOutputCount();
    for (size_t i = 0; i < num_outputs; ++i) {
        output_name_ptrs.emplace_back(session.GetOutputNameAllocated(i, Ort::AllocatorWithDefaultOptions()));
        output_names.push_back(output_name_ptrs.back().get());
    }

    // Abrir video
    VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        cerr << "No se pudo abrir el video" << endl;
        return -1;
    }

    int frame_width = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));

    namedWindow("Detección MobileNetV3", WINDOW_AUTOSIZE);

    // Variables para cálculo de FPS
    auto start_time = high_resolution_clock::now();
    int frame_count = 0;

    Mat frame;
    while (cap.read(frame)) {
        auto frame_start = high_resolution_clock::now();

        // Preprocesamiento de la imagen
        Mat input_tensor_image = preprocess(frame, 320, 320);

        // Crear tensor de entrada
        vector<uint8_t> input_tensor_values(input_tensor_image.total() * input_tensor_image.channels());
        memcpy(input_tensor_values.data(), input_tensor_image.data, input_tensor_values.size() * sizeof(uint8_t));

        vector<int64_t> input_tensor_shape = {1, 320, 320, 3}; // Formato NHWC

        Ort::Value input_tensor = Ort::Value::CreateTensor<uint8_t>(
            memory_info, input_tensor_values.data(), input_tensor_values.size(),
            input_tensor_shape.data(), input_tensor_shape.size());

        // Ejecutar inferencia
        auto output_tensors = session.Run(Ort::RunOptions{nullptr},
                                          input_names.data(), &input_tensor, 1,
                                          output_names.data(), output_names.size());

        // Obtener resultados
        float* boxes = output_tensors[0].GetTensorMutableData<float>();
        float* labels = output_tensors[1].GetTensorMutableData<float>();
        float* scores = output_tensors[2].GetTensorMutableData<float>();

        auto shape_boxes = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        int num_detections = static_cast<int>(shape_boxes[0]);

        // Dibujar resultados
        for (int i = 0; i < num_detections; ++i) {
            float score = scores[i];
            if (score < 0.3f)
                continue;

            int class_id = static_cast<int>(labels[i]) - 1;
            string label_text = (class_id >= 0 && class_id < (int)class_names.size()) ? class_names[class_id] : "ID:" + to_string(class_id + 1);

            float ymin = boxes[i * 4 + 0] * frame_height;
            float xmin = boxes[i * 4 + 1] * frame_width;
            float ymax = boxes[i * 4 + 2] * frame_height;
            float xmax = boxes[i * 4 + 3] * frame_width;

            Rect rect(Point((int)xmin, (int)ymin), Point((int)xmax, (int)ymax));
            rectangle(frame, rect, Scalar(0, 255, 0), 2);

            string text = label_text + " " + to_string(int(score * 100)) + "%";
            int baseLine;
            Size label_size = getTextSize(text, FONT_HERSHEY_SIMPLEX, 0.3, 1, &baseLine);
            int top = max((int)ymin, label_size.height);

            // Fondo del texto
            rectangle(frame, Point((int)xmin, top - label_size.height),
                      Point((int)xmin + label_size.width, top + baseLine),
                      Scalar(0, 255, 0), FILLED);
            putText(frame, text, Point((int)xmin, top), FONT_HERSHEY_SIMPLEX, 0.3, Scalar(0, 0, 0), 1);
        }

        // Calcular y mostrar FPS
        frame_count++;
        auto now = high_resolution_clock::now();
        double elapsed = duration_cast<duration<double>>(now - start_time).count();
        double fps = frame_count / elapsed;

        string fps_text = "FPS: " + to_string(int(fps));
        putText(frame, fps_text, Point(20, 40), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255, 0, 0), 2);

        imshow("Detección MobileNetV3", frame);

        // Salir si se presiona ESC
        if (waitKey(1) == 27) break;
    }

    // Liberar recursos
    cap.release();
    destroyAllWindows();

    return 0;
}
