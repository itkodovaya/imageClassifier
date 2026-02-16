#ifndef LANGUAGE_MODEL_H
#define LANGUAGE_MODEL_H

#include <Eigen/Dense>
#include <vector>
#include <string>
#include <map>
#include <memory>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <locale>
#include <codecvt>

using namespace Eigen;
using namespace std;

// Структура для хранения сообщения в чате
struct ChatMessage {
    string role;  // "user" или "assistant"
    string content;
    string language;  // "ru" или "en"
};

// Класс для токенизации текста
class Tokenizer {
public:
    Tokenizer();
    
    // Токенизация текста в последовательность индексов
    vector<int> encode(const string& text);
    
    // Декодирование последовательности индексов в текст
    string decode(const vector<int>& tokens);
    
    // Определение языка текста
    string detectLanguage(const string& text);
    
    // Получение размера словаря
    int getVocabSize() const { return vocab_size; }
    
    // Загрузка словаря из файла
    bool loadVocabulary(const string& path);
    
    // Сохранение словаря в файл
    bool saveVocabulary(const string& path);
    
    // Добавление новых слов в словарь
    void addToVocabulary(const string& text);
    
    // BPE (Byte Pair Encoding) функции
    void trainBPE(const vector<string>& texts, int num_merges = 1000);
    vector<string> applyBPE(const string& text);
    void loadBPE(const string& path);
    void saveBPE(const string& path);
    
    // Использование BPE
    bool use_bpe;
    map<string, int> bpe_merges;  // Пары символов -> приоритет слияния
    vector<pair<string, string>> bpe_vocab;  // Словарь BPE

private:
    map<string, int> word_to_id;
    map<int, string> id_to_word;
    int vocab_size;
    int next_id;
    
    // Разделение текста на слова с учетом русского и английского
    vector<string> tokenize(const string& text);
    
    // Нормализация текста
    string normalize(const string& text);
};

// Класс языковой модели (простая RNN)
class LanguageModel {
public:
    LanguageModel(int vocab_size, int hidden_size = 256, int num_layers = 2, int num_heads = 4, int num_transformer_layers = 1);
    
    // Генерация ответа на запрос (с LSTM и Attention)
    string generateResponse(const string& user_input, const vector<ChatMessage>& context = {});
    string generateResponseBeamSearch(const string& user_input, const vector<ChatMessage>& context = {}, int beam_width = 3);
    string filterResponseByLanguage(const string& response, const string& target_language);
    
    // Обучение модели на диалогах
    void trainOnDialogues(const vector<vector<ChatMessage>>& dialogues, int epochs = 10, int batch_size = 1,
                         double validation_split = 0.1, int patience = 5, const string& checkpoint_dir = "");
    
    // Сохранение модели
    void saveModel(const string& path);
    
    // Загрузка модели
    void loadModel(const string& path);
    
    // Квантизация модели
    void quantizeModel(int bits = 8);  // Квантизация в INT8
    void dequantizeModel();  // Возврат к float64
    bool isQuantized() const { return model_quantized; }
    
    // Early stopping и checkpointing
    bool shouldStopEarly(double current_val_loss, int& no_improvement_count, double best_loss, int patience);
    void saveCheckpoint(const string& path, int epoch, double loss);
    bool loadCheckpoint(const string& path, int& epoch, double& loss);
    
    // Data Augmentation
    vector<vector<ChatMessage>> augmentDialogues(const vector<vector<ChatMessage>>& dialogues);
    vector<ChatMessage> augmentDialogue(const vector<ChatMessage>& dialogue);
    string paraphrase(const string& text);
    string backTranslate(const string& text);
    
    // Установка параметров генерации
    void setTemperature(double temp) { temperature = temp; }
    void setMaxLength(int len) { max_length = len; }
    void setTopK(int k) { top_k = k; }
    void setTopP(double p) { top_p = p; }
    void setRepetitionPenalty(double penalty) { repetition_penalty = penalty; }

private:
    int vocab_size;
    int hidden_size;
    int num_layers;
    int num_heads;  // Количество голов внимания для Transformer
    int d_model;    // Размерность модели для Transformer
    int num_transformer_layers;  // Количество слоев Transformer
    
    // Encoder-Decoder архитектура
    bool use_encoder_decoder;  // Использовать encoder-decoder архитектуру
    int encoder_layers;  // Количество слоев encoder
    int decoder_layers;  // Количество слоев decoder
    
    double temperature;
    int max_length;
    int top_k;  // Top-K sampling: выбираем только из K наиболее вероятных токенов
    double top_p;  // Top-P (nucleus) sampling: выбираем из токенов с кумулятивной вероятностью P
    double repetition_penalty;  // Штраф за повторения
    double dropout_rate;  // Dropout для регуляризации
    double gradient_clip;  // Максимальное значение градиента (gradient clipping)
    double length_penalty;  // Штраф за длину ответа
    double diversity_penalty;  // Штраф за разнообразие
    int context_window;  // Максимальный размер контекста
    
    // Early stopping параметры
    double best_validation_loss;
    int no_improvement_epochs;
    string checkpoint_directory;
    
    // Mixed precision training
    bool use_mixed_precision;
    
    // Gradient accumulation
    int gradient_accumulation_steps;
    
    // Параллелизация
    int num_threads;  // Количество потоков для параллельных вычислений
    
    // Квантизация
    bool model_quantized;  // Флаг квантизации модели
    double quantization_scale;  // Масштаб для квантизации
    
    // Оптимизация вычислений
    bool use_optimizations;  // Использовать оптимизированные матричные операции
    
    // Adam optimizer параметры
    vector<MatrixXd> m_W_f, m_W_i, m_W_c, m_W_o_lstm;  // Моменты для Adam
    vector<MatrixXd> v_W_f, v_W_i, v_W_c, v_W_o_lstm;
    vector<MatrixXd> m_U_f, m_U_i, m_U_c, m_U_o;
    vector<MatrixXd> v_U_f, v_U_i, v_U_c, v_U_o;
    vector<VectorXd> m_b_f, m_b_i, m_b_c, m_b_o_lstm;
    vector<VectorXd> v_b_f, v_b_i, v_b_c, v_b_o_lstm;
    MatrixXd m_W_hy, v_W_hy;
    VectorXd m_b_y, v_b_y;
    MatrixXd m_embeddings, v_embeddings;
    int adam_step;  // Счетчик шагов для Adam
    
    // Beam search параметры
    int beam_width;  // Ширина луча для beam search
    
    // Positional encoding
    MatrixXd positional_encoding;
    
    // Метрики
    double current_perplexity;
    double current_bleu_score;
    vector<double> training_losses;  // История потерь при обучении
    vector<double> validation_losses;  // История потерь на валидации
    map<string, vector<double>> metrics_history;  // История метрик для мониторинга
    
    // Веса LSTM (вместо простой RNN)
    // Для каждого слоя LSTM: 4 ворот (forget, input, candidate, output)
    vector<MatrixXd> W_f;   // Forget gate веса (вход->forget)
    vector<MatrixXd> W_i;   // Input gate веса (вход->input)
    vector<MatrixXd> W_c;   // Candidate gate веса (вход->candidate)
    vector<MatrixXd> W_o_lstm;   // Output gate веса (вход->output) для LSTM
    vector<MatrixXd> U_f;   // Forget gate веса (скрытый->forget)
    vector<MatrixXd> U_i;   // Input gate веса (скрытый->input)
    vector<MatrixXd> U_c;   // Candidate gate веса (скрытый->candidate)
    vector<MatrixXd> U_o;   // Output gate веса (скрытый->output)
    vector<VectorXd> b_f;   // Forget gate смещения
    vector<VectorXd> b_i;   // Input gate смещения
    vector<VectorXd> b_c;   // Candidate gate смещения
    vector<VectorXd> b_o_lstm;   // Output gate смещения для LSTM
    
    // Веса для выходного слоя
    vector<MatrixXd> W_hy;  // Веса скрытый->выход
    vector<VectorXd> b_y;   // Смещения выходного слоя
    
    // Encoder-Decoder веса (отдельные веса для encoder и decoder)
    vector<MatrixXd> W_f_encoder;   // Encoder: Forget gate веса
    vector<MatrixXd> W_i_encoder;   // Encoder: Input gate веса
    vector<MatrixXd> W_c_encoder;   // Encoder: Candidate gate веса
    vector<MatrixXd> W_o_encoder;   // Encoder: Output gate веса
    vector<MatrixXd> U_f_encoder;   // Encoder: Forget gate веса (скрытый)
    vector<MatrixXd> U_i_encoder;   // Encoder: Input gate веса (скрытый)
    vector<MatrixXd> U_c_encoder;   // Encoder: Candidate gate веса (скрытый)
    vector<MatrixXd> U_o_encoder;   // Encoder: Output gate веса (скрытый)
    vector<VectorXd> b_f_encoder;   // Encoder: Forget gate смещения
    vector<VectorXd> b_i_encoder;   // Encoder: Input gate смещения
    vector<VectorXd> b_c_encoder;   // Encoder: Candidate gate смещения
    vector<VectorXd> b_o_encoder;   // Encoder: Output gate смещения
    
    vector<MatrixXd> W_f_decoder;   // Decoder: Forget gate веса
    vector<MatrixXd> W_i_decoder;   // Decoder: Input gate веса
    vector<MatrixXd> W_c_decoder;   // Decoder: Candidate gate веса
    vector<MatrixXd> W_o_decoder;   // Decoder: Output gate веса
    vector<MatrixXd> U_f_decoder;   // Decoder: Forget gate веса (скрытый)
    vector<MatrixXd> U_i_decoder;   // Decoder: Input gate веса (скрытый)
    vector<MatrixXd> U_c_decoder;   // Decoder: Candidate gate веса (скрытый)
    vector<MatrixXd> U_o_decoder;   // Decoder: Output gate веса (скрытый)
    vector<VectorXd> b_f_decoder;   // Decoder: Forget gate смещения
    vector<VectorXd> b_i_decoder;   // Decoder: Input gate смещения
    vector<VectorXd> b_c_decoder;   // Decoder: Candidate gate смещения
    vector<VectorXd> b_o_decoder;   // Decoder: Output gate смещения
    
    // Cross-Attention веса (decoder обращает внимание на encoder outputs)
    MatrixXd W_q_cross;  // Query веса для cross-attention
    MatrixXd W_k_cross;  // Key веса для cross-attention
    MatrixXd W_v_cross;  // Value веса для cross-attention
    MatrixXd W_o_cross;  // Output веса для cross-attention
    
    // Attention механизм
    MatrixXd W_attention;   // Веса для вычисления attention scores
    VectorXd b_attention;  // Смещения для attention
    
    // Transformer компоненты (упрощенная версия)
    // Для каждого слоя Transformer: веса для каждой головы внимания
    vector<vector<MatrixXd>> W_q_layers;  // Query веса для каждого слоя и каждой головы
    vector<vector<MatrixXd>> W_k_layers;  // Key веса для каждого слоя и каждой головы
    vector<vector<MatrixXd>> W_v_layers;  // Value веса для каждого слоя и каждой головы
    vector<vector<MatrixXd>> W_o_transformer_layers;  // Output веса для каждого слоя и каждой головы
    vector<MatrixXd> W_ff1_layers;  // Feed-forward слой 1 для каждого слоя Transformer
    vector<MatrixXd> W_ff2_layers;  // Feed-forward слой 2 для каждого слоя Transformer
    vector<VectorXd> b_ff1_layers;  // Смещение FF слоя 1 для каждого слоя Transformer
    vector<VectorXd> b_ff2_layers;  // Смещение FF слоя 2 для каждого слоя Transformer
    
    // Обратная совместимость (для первого слоя)
    vector<MatrixXd> W_q;  // Query веса для каждой головы (первый слой)
    vector<MatrixXd> W_k;  // Key веса для каждой головы (первый слой)
    vector<MatrixXd> W_v;  // Value веса для каждой головы (первый слой)
    vector<MatrixXd> W_o_transformer;  // Output веса для каждой головы Transformer (первый слой)
    MatrixXd W_ff1;  // Feed-forward слой 1 (первый слой)
    MatrixXd W_ff2;  // Feed-forward слой 2 (первый слой)
    VectorXd b_ff1;  // Смещение FF слоя 1 (первый слой)
    VectorXd b_ff2;  // Смещение FF слоя 2 (первый слой)
    
    // Встроенные представления слов
    MatrixXd embeddings;
    
    // Прямое распространение через LSTM
    vector<VectorXd> forwardLSTM(const vector<int>& tokens);
    
    // Параллельная версия forwardLSTM
    vector<VectorXd> forwardLSTMParallel(const vector<int>& tokens);
    
    // Encoder-Decoder архитектура
    vector<VectorXd> forwardEncoder(const vector<int>& tokens);
    vector<VectorXd> forwardDecoder(const vector<int>& tokens, const vector<VectorXd>& encoder_outputs);
    
    // LSTM cell forward pass
    void lstmCellForward(const VectorXd& x, const VectorXd& h_prev, const VectorXd& c_prev,
                        int layer, VectorXd& h_out, VectorXd& c_out, bool training = false);
    
    // Attention механизм
    VectorXd computeAttention(const vector<VectorXd>& hidden_states, const VectorXd& current_hidden);
    
    // Multi-Head Attention (Transformer)
    VectorXd multiHeadAttention(const vector<VectorXd>& hidden_states, const VectorXd& current_hidden, int layer_idx = 0);
    
    // Feed-Forward Network (Transformer)
    VectorXd feedForward(const VectorXd& x, int layer_idx = 0);
    
    // Layer Normalization (упрощенная версия)
    VectorXd layerNorm(const VectorXd& x);
    
    // Transformer блок (Self-Attention + FFN)
    VectorXd transformerBlock(const vector<VectorXd>& hidden_states, const VectorXd& current_hidden, int layer_idx = 0);
    
    // Обратное распространение (Backpropagation Through Time для LSTM)
    void backward(const vector<int>& tokens, const vector<int>& target_tokens, 
                 const vector<VectorXd>& hidden_states, const vector<VectorXd>& cell_states);
    
    // Вычисление градиентов для одного шага LSTM
    void lstmCellBackward(const VectorXd& x, const VectorXd& h_prev, const VectorXd& c_prev,
                         const VectorXd& h_out, const VectorXd& c_out,
                         const VectorXd& dh_next, const VectorXd& dc_next,
                         int layer, VectorXd& dx, VectorXd& dh_prev, VectorXd& dc_prev);
    
    // Обновление весов с использованием градиентов
    void updateWeights(double learning_rate);
    void updateWeightsAdam(double learning_rate, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8);
    
    // Генерация следующего токена
    int sampleNextToken(const VectorXd& logits, const vector<int>& previous_tokens = {});
    
    // Инициализация
    void initializeWeights();
    void initializeAdam();
    void initializePositionalEncoding();
    
    // Dropout
    VectorXd applyDropout(const VectorXd& x, bool training = true);
    MatrixXd applyDropout(const MatrixXd& x, bool training = true);
    
    // Метрики
    double computePerplexity(const vector<int>& tokens, const vector<VectorXd>& hidden_states);
    double computeBLEUScore(const string& generated, const string& reference);
    double computeROUGEScore(const string& generated, const string& reference, int n = 1);
    double computeMETEORScore(const string& generated, const string& reference);
    
    // Функции активации
    VectorXd tanh(const VectorXd& x);
    VectorXd sigmoid(const VectorXd& x);
    VectorXd softmax(const VectorXd& x);
    VectorXd relu(const VectorXd& x);
    
    // Вычисление потерь
    double computeLoss(const VectorXd& logits, int target_token);
    
    // Простая генерация ответа (fallback)
    string generateSimpleResponse(const string& user_input);
    
    // Toxicity Detection
    bool isToxic(const string& text);
    double getToxicityScore(const string& text);
    string filterToxicContent(const string& text);
    
    // Monitoring
    void logTrainingMetrics(int epoch, double train_loss, double val_loss);
    void saveMetricsToFile(const string& path);
    
    // Оптимизация вычислений (BLAS-подобные оптимизации)
    VectorXd optimizedMatrixVectorMultiply(const MatrixXd& A, const VectorXd& x);
    MatrixXd optimizedMatrixMatrixMultiply(const MatrixXd& A, const MatrixXd& B);
    void enableOptimizations(bool enable) { use_optimizations = enable; }
    
    // Хранение градиентов для обновления весов
    vector<MatrixXd> dW_f, dW_i, dW_c, dW_o_lstm;
    vector<MatrixXd> dU_f, dU_i, dU_c, dU_o;
    vector<VectorXd> db_f, db_i, db_c, db_o_lstm;
    vector<MatrixXd> dW_hy;
    vector<VectorXd> db_y;
    MatrixXd dembeddings;
};

#endif // LANGUAGE_MODEL_H

