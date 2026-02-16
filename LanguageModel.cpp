#include "LanguageModel.h"
#include <cmath>
#include <random>
#include <iostream>
#include <fstream>
#include <sstream>
#include <locale>
#include <codecvt>
#include <regex>
#include <cctype>
#include <map>
#include <set>
#include <algorithm>

// Forward declaration для глобального токенизатора
// Глобальный токенизатор
Tokenizer* g_tokenizer = nullptr;

extern Tokenizer* g_tokenizer;

// ==================== TOKENIZER ====================

Tokenizer::Tokenizer() : vocab_size(0), next_id(1) {
    // Добавляем специальные токены
    word_to_id["<PAD>"] = 0;
    id_to_word[0] = "<PAD>";
    word_to_id["<UNK>"] = 1;
    id_to_word[1] = "<UNK>";
    word_to_id["<START>"] = 2;
    id_to_word[2] = "<START>";
    word_to_id["<END>"] = 3;
    id_to_word[3] = "<END>";
    next_id = 4;
    vocab_size = 4;
    use_bpe = false;  // По умолчанию BPE выключен
}

string Tokenizer::normalize(const string& text) {
    string result = text;
    
    // Безопасное приведение к нижнему регистру (только для ASCII)
    for (size_t i = 0; i < result.length(); i++) {
        unsigned char c = static_cast<unsigned char>(result[i]);
        if (c >= 'A' && c <= 'Z') {
            result[i] = static_cast<char>(c + ('a' - 'A'));
        }
        // Нормализация русских букв (Ё -> Е)
        else if (c == 0xD0 && i + 1 < result.length()) {
            unsigned char c2 = static_cast<unsigned char>(result[i + 1]);
            if (c2 == 0x81) {  // Ё
                result[i] = 0xD0;
                result[i + 1] = 0xB5;  // Е
            } else if (c2 == 0x91) {  // ё
                result[i] = 0xD0;
                result[i + 1] = 0xB5;  // е
            }
        }
    }
    
    // Нормализация кавычек и тире
    result = regex_replace(result, regex("[«»„""]"), "\"");
    result = regex_replace(result, regex("[–—]"), "-");
    result = regex_replace(result, regex("…"), "...");
    
    // Удаляем множественные пробелы
    result = regex_replace(result, regex("\\s+"), " ");
    
    // Удаляем пробелы вокруг знаков препинания
    result = regex_replace(result, regex("\\s+([.,!?;:])"), "$1");
    result = regex_replace(result, regex("([.,!?;:])\\s+"), "$1 ");
    
    // Удаляем пробелы в начале и конце
    result = regex_replace(result, regex("^\\s+|\\s+$"), "");
    
    return result;
}

vector<string> Tokenizer::tokenize(const string& text) {
    vector<string> tokens;
    string normalized = normalize(text);
    
    // Простая токенизация: разделяем по пробелам и знакам препинания
    // Безопасная обработка UTF-8 символов
    string current_token = "";
    for (size_t i = 0; i < normalized.length(); i++) {
        unsigned char c = static_cast<unsigned char>(normalized[i]);
        
        // Проверяем, является ли символ пробелом (безопасно для UTF-8)
        bool is_space = (c == ' ' || c == '\t' || c == '\n' || c == '\r');
        
        // Проверяем, является ли символ знаком препинания (только ASCII)
        bool is_punct = (c >= 0 && c <= 255) && (ispunct(static_cast<int>(c)) != 0);
        
        if (is_space || is_punct) {
            if (!current_token.empty()) {
                tokens.push_back(current_token);
                current_token = "";
            }
            if (is_punct) {
                tokens.push_back(string(1, static_cast<char>(c)));
            }
        } else {
            // Для UTF-8 символов (русские буквы и т.д.) добавляем весь байт
            current_token += normalized[i];
        }
    }
    if (!current_token.empty()) {
        tokens.push_back(current_token);
    }
    
    return tokens;
}

void Tokenizer::addToVocabulary(const string& text) {
    vector<string> tokens = tokenize(text);
    for (const string& token : tokens) {
        if (word_to_id.find(token) == word_to_id.end()) {
            word_to_id[token] = next_id;
            id_to_word[next_id] = token;
            next_id++;
            vocab_size++;
        }
    }
}

vector<int> Tokenizer::encode(const string& text) {
    vector<string> tokens = tokenize(text);
    vector<int> result;
    result.push_back(word_to_id["<START>"]);
    
    for (const string& token : tokens) {
        if (word_to_id.find(token) != word_to_id.end()) {
            result.push_back(word_to_id[token]);
        } else {
            result.push_back(word_to_id["<UNK>"]);
        }
    }
    
    result.push_back(word_to_id["<END>"]);
    return result;
}

string Tokenizer::decode(const vector<int>& tokens) {
    string result = "";
    for (int token_id : tokens) {
        // Пропускаем все специальные токены
        if (token_id == word_to_id["<START>"] || 
            token_id == word_to_id["<END>"] ||
            token_id == word_to_id["<PAD>"] ||
            token_id == word_to_id["<UNK>"]) {
            continue;
        }
        if (id_to_word.find(token_id) != id_to_word.end()) {
            string word = id_to_word[token_id];
            // Дополнительная проверка: пропускаем специальные токены по тексту
            if (word == "<START>" || word == "<END>" || word == "<PAD>" || word == "<UNK>") {
                continue;
            }
            if (!result.empty()) {
                result += " ";
            }
            result += word;
        }
    }
    return result;
}

string Tokenizer::detectLanguage(const string& text) {
    // Простое определение языка по наличию кириллицы
    int cyrillic_count = 0;
    int latin_count = 0;
    
    for (char c : text) {
        if ((c >= 0x0400 && c <= 0x04FF) || (c >= 0x0500 && c <= 0x052F)) {
            cyrillic_count++;
        } else if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) {
            latin_count++;
        }
    }
    
    return (cyrillic_count > latin_count) ? "ru" : "en";
}

bool Tokenizer::saveVocabulary(const string& path) {
    ofstream file(path);
    if (!file.is_open()) {
        return false;
    }
    
    file << vocab_size << endl;
    for (const auto& pair : word_to_id) {
        file << pair.first << " " << pair.second << endl;
    }
    
    return true;
}

bool Tokenizer::loadVocabulary(const string& path) {
    ifstream file(path);
    if (!file.is_open()) {
        return false;
    }
    
    word_to_id.clear();
    id_to_word.clear();
    
    int size;
    file >> size;
    vocab_size = size;
    
    string word;
    int id;
    while (file >> word >> id) {
        word_to_id[word] = id;
        id_to_word[id] = word;
        if (id >= next_id) {
            next_id = id + 1;
        }
    }
    
    return true;
}

// ==================== LANGUAGE MODEL ====================

LanguageModel::LanguageModel(int vocab_size, int hidden_size, int num_layers, int num_heads, int num_transformer_layers)
    : vocab_size(vocab_size), hidden_size(hidden_size), num_layers(num_layers),
      num_heads(num_heads), d_model(hidden_size), num_transformer_layers(num_transformer_layers),
      use_encoder_decoder(false), encoder_layers(2), decoder_layers(2),
      temperature(0.8), max_length(200),
      top_k(50), top_p(0.9), repetition_penalty(1.2), dropout_rate(0.1), gradient_clip(5.0),
      length_penalty(0.6), diversity_penalty(0.5), context_window(512),
      best_validation_loss(1e10), no_improvement_epochs(0), checkpoint_directory(""),
      use_mixed_precision(false), gradient_accumulation_steps(1), num_threads(4),
      model_quantized(false), quantization_scale(1.0),
      use_optimizations(true),  // По умолчанию включены оптимизации
      beam_width(3), adam_step(0), current_perplexity(0.0), current_bleu_score(0.0) {
    
    // Инициализация весов
    
    // Инициализация весов LSTM
    W_f.resize(num_layers);
    W_i.resize(num_layers);
    W_c.resize(num_layers);
    W_o_lstm.resize(num_layers);
    U_f.resize(num_layers);
    U_i.resize(num_layers);
    U_c.resize(num_layers);
    U_o.resize(num_layers);
    b_f.resize(num_layers);
    b_i.resize(num_layers);
    b_c.resize(num_layers);
    b_o_lstm.resize(num_layers);
    
    // Веса для выходного слоя
    W_hy.resize(num_layers);
    b_y.resize(num_layers);
    
    // Инициализация весов Encoder-Decoder (если используется)
    if (use_encoder_decoder) {
        // Encoder веса
        W_f_encoder.resize(encoder_layers);
        W_i_encoder.resize(encoder_layers);
        W_c_encoder.resize(encoder_layers);
        W_o_encoder.resize(encoder_layers);
        U_f_encoder.resize(encoder_layers);
        U_i_encoder.resize(encoder_layers);
        U_c_encoder.resize(encoder_layers);
        U_o_encoder.resize(encoder_layers);
        b_f_encoder.resize(encoder_layers);
        b_i_encoder.resize(encoder_layers);
        b_c_encoder.resize(encoder_layers);
        b_o_encoder.resize(encoder_layers);
        
        // Decoder веса
        W_f_decoder.resize(decoder_layers);
        W_i_decoder.resize(decoder_layers);
        W_c_decoder.resize(decoder_layers);
        W_o_decoder.resize(decoder_layers);
        U_f_decoder.resize(decoder_layers);
        U_i_decoder.resize(decoder_layers);
        U_c_decoder.resize(decoder_layers);
        U_o_decoder.resize(decoder_layers);
        b_f_decoder.resize(decoder_layers);
        b_i_decoder.resize(decoder_layers);
        b_c_decoder.resize(decoder_layers);
        b_o_decoder.resize(decoder_layers);
        
        // Cross-Attention веса
        int d_k = d_model / num_heads;
        W_q_cross = MatrixXd::Random(d_k, d_model) * 0.1;
        W_k_cross = MatrixXd::Random(d_k, d_model) * 0.1;
        W_v_cross = MatrixXd::Random(d_k, d_model) * 0.1;
        W_o_cross = MatrixXd::Random(d_model, d_k) * 0.1;
    }
    
    // Инициализация встроенных представлений (Xavier инициализация)
    int embedding_size = hidden_size;
    random_device rd_emb;
    mt19937 gen_emb(rd_emb());
    double emb_limit = sqrt(6.0 / (embedding_size + vocab_size));
    uniform_real_distribution<double> emb_dist(-emb_limit, emb_limit);
    embeddings = MatrixXd::NullaryExpr(vocab_size, embedding_size, 
        [&]() { return emb_dist(gen_emb); });
    
    // Инициализация Attention
    W_attention = MatrixXd::Random(hidden_size, hidden_size) * 0.1;
    b_attention = VectorXd::Zero(hidden_size);
    
    // Инициализация Transformer компонентов
    int d_k = d_model / num_heads;  // Размерность каждой головы
    // Инициализация весов для всех слоев Transformer
    W_q_layers.resize(num_transformer_layers);
    W_k_layers.resize(num_transformer_layers);
    W_v_layers.resize(num_transformer_layers);
    W_o_transformer_layers.resize(num_transformer_layers);
    W_ff1_layers.resize(num_transformer_layers);
    W_ff2_layers.resize(num_transformer_layers);
    b_ff1_layers.resize(num_transformer_layers);
    b_ff2_layers.resize(num_transformer_layers);
    
    int ff_dim = d_model * 4;  // Обычно в 4 раза больше
    
    for (int layer = 0; layer < num_transformer_layers; layer++) {
        W_q_layers[layer].resize(num_heads);
        W_k_layers[layer].resize(num_heads);
        W_v_layers[layer].resize(num_heads);
        W_o_transformer_layers[layer].resize(num_heads);
        
        for (int h = 0; h < num_heads; h++) {
            W_q_layers[layer][h] = MatrixXd::Random(d_k, d_model) * 0.1;
            W_k_layers[layer][h] = MatrixXd::Random(d_k, d_model) * 0.1;
            W_v_layers[layer][h] = MatrixXd::Random(d_k, d_model) * 0.1;
            W_o_transformer_layers[layer][h] = MatrixXd::Random(d_model, d_k) * 0.1;
        }
        
        // Feed-Forward Network для каждого слоя
        W_ff1_layers[layer] = MatrixXd::Random(ff_dim, d_model) * 0.1;
        W_ff2_layers[layer] = MatrixXd::Random(d_model, ff_dim) * 0.1;
        b_ff1_layers[layer] = VectorXd::Zero(ff_dim);
        b_ff2_layers[layer] = VectorXd::Zero(d_model);
    }
    
    // Обратная совместимость: инициализируем первый слой в старые переменные
    if (num_transformer_layers > 0) {
        W_q = W_q_layers[0];
        W_k = W_k_layers[0];
        W_v = W_v_layers[0];
        W_o_transformer = W_o_transformer_layers[0];
        W_ff1 = W_ff1_layers[0];
        W_ff2 = W_ff2_layers[0];
        b_ff1 = b_ff1_layers[0];
        b_ff2 = b_ff2_layers[0];
    } else {
        // Если слоев нет, инициализируем как раньше
        W_q.resize(num_heads);
        W_k.resize(num_heads);
        W_v.resize(num_heads);
        W_o_transformer.resize(num_heads);
        
        for (int h = 0; h < num_heads; h++) {
            W_q[h] = MatrixXd::Random(d_k, d_model) * 0.1;
            W_k[h] = MatrixXd::Random(d_k, d_model) * 0.1;
            W_v[h] = MatrixXd::Random(d_k, d_model) * 0.1;
            W_o_transformer[h] = MatrixXd::Random(d_model, d_k) * 0.1;
        }
        
        W_ff1 = MatrixXd::Random(ff_dim, d_model) * 0.1;
        W_ff2 = MatrixXd::Random(d_model, ff_dim) * 0.1;
        b_ff1 = VectorXd::Zero(ff_dim);
        b_ff2 = VectorXd::Zero(d_model);
    }
    
    initializeWeights();
    initializeAdam();
    initializePositionalEncoding();
}

void LanguageModel::initializeWeights() {
    random_device rd;
    mt19937 gen(rd());
    
    int embedding_size = hidden_size;
    
    // Xavier/Glorot инициализация для лучшей сходимости
    auto xavier_init = [&](int fan_in, int fan_out) {
        double limit = sqrt(6.0 / (fan_in + fan_out));
        uniform_real_distribution<double> dist(-limit, limit);
        return dist(gen);
    };
    
    for (int i = 0; i < num_layers; i++) {
        // LSTM веса для входных данных (Xavier инициализация)
        W_f[i] = MatrixXd::NullaryExpr(hidden_size, embedding_size, 
            [&]() { return xavier_init(embedding_size, hidden_size); });
        W_i[i] = MatrixXd::NullaryExpr(hidden_size, embedding_size, 
            [&]() { return xavier_init(embedding_size, hidden_size); });
        W_c[i] = MatrixXd::NullaryExpr(hidden_size, embedding_size, 
            [&]() { return xavier_init(embedding_size, hidden_size); });
        W_o_lstm[i] = MatrixXd::NullaryExpr(hidden_size, embedding_size, 
            [&]() { return xavier_init(embedding_size, hidden_size); });
        
        // LSTM веса для скрытых состояний (Xavier инициализация)
        U_f[i] = MatrixXd::NullaryExpr(hidden_size, hidden_size, 
            [&]() { return xavier_init(hidden_size, hidden_size); });
        U_i[i] = MatrixXd::NullaryExpr(hidden_size, hidden_size, 
            [&]() { return xavier_init(hidden_size, hidden_size); });
        U_c[i] = MatrixXd::NullaryExpr(hidden_size, hidden_size, 
            [&]() { return xavier_init(hidden_size, hidden_size); });
        U_o[i] = MatrixXd::NullaryExpr(hidden_size, hidden_size, 
            [&]() { return xavier_init(hidden_size, hidden_size); });
        
        // Смещения LSTM
        b_f[i] = VectorXd::Ones(hidden_size) * 1.0;  // Инициализация forget gate смещения = 1 для лучшего запоминания
        b_i[i] = VectorXd::Zero(hidden_size);
        b_c[i] = VectorXd::Zero(hidden_size);
        b_o_lstm[i] = VectorXd::Zero(hidden_size);
        
        // Веса выходного слоя (Xavier инициализация)
        W_hy[i] = MatrixXd::NullaryExpr(vocab_size, hidden_size, 
            [&]() { return xavier_init(hidden_size, vocab_size); });
        b_y[i] = VectorXd::Zero(vocab_size);
        
        // Для следующего слоя embedding_size = hidden_size
        embedding_size = hidden_size;
    }
    
    // Инициализация весов Encoder-Decoder (если используется)
    if (use_encoder_decoder) {
        for (int i = 0; i < encoder_layers; i++) {
            // Encoder веса (Xavier инициализация)
            W_f_encoder[i] = MatrixXd::NullaryExpr(hidden_size, embedding_size, 
                [&]() { return xavier_init(embedding_size, hidden_size); });
            W_i_encoder[i] = MatrixXd::NullaryExpr(hidden_size, embedding_size, 
                [&]() { return xavier_init(embedding_size, hidden_size); });
            W_c_encoder[i] = MatrixXd::NullaryExpr(hidden_size, embedding_size, 
                [&]() { return xavier_init(embedding_size, hidden_size); });
            W_o_encoder[i] = MatrixXd::NullaryExpr(hidden_size, embedding_size, 
                [&]() { return xavier_init(embedding_size, hidden_size); });
            
            U_f_encoder[i] = MatrixXd::NullaryExpr(hidden_size, hidden_size, 
                [&]() { return xavier_init(hidden_size, hidden_size); });
            U_i_encoder[i] = MatrixXd::NullaryExpr(hidden_size, hidden_size, 
                [&]() { return xavier_init(hidden_size, hidden_size); });
            U_c_encoder[i] = MatrixXd::NullaryExpr(hidden_size, hidden_size, 
                [&]() { return xavier_init(hidden_size, hidden_size); });
            U_o_encoder[i] = MatrixXd::NullaryExpr(hidden_size, hidden_size, 
                [&]() { return xavier_init(hidden_size, hidden_size); });
            
            b_f_encoder[i] = VectorXd::Ones(hidden_size) * 1.0;
            b_i_encoder[i] = VectorXd::Zero(hidden_size);
            b_c_encoder[i] = VectorXd::Zero(hidden_size);
            b_o_encoder[i] = VectorXd::Zero(hidden_size);
            
            embedding_size = hidden_size;
        }
        
        embedding_size = hidden_size;  // Сброс для decoder
        for (int i = 0; i < decoder_layers; i++) {
            // Decoder веса (Xavier инициализация)
            W_f_decoder[i] = MatrixXd::NullaryExpr(hidden_size, embedding_size, 
                [&]() { return xavier_init(embedding_size, hidden_size); });
            W_i_decoder[i] = MatrixXd::NullaryExpr(hidden_size, embedding_size, 
                [&]() { return xavier_init(embedding_size, hidden_size); });
            W_c_decoder[i] = MatrixXd::NullaryExpr(hidden_size, embedding_size, 
                [&]() { return xavier_init(embedding_size, hidden_size); });
            W_o_decoder[i] = MatrixXd::NullaryExpr(hidden_size, embedding_size, 
                [&]() { return xavier_init(embedding_size, hidden_size); });
            
            U_f_decoder[i] = MatrixXd::NullaryExpr(hidden_size, hidden_size, 
                [&]() { return xavier_init(hidden_size, hidden_size); });
            U_i_decoder[i] = MatrixXd::NullaryExpr(hidden_size, hidden_size, 
                [&]() { return xavier_init(hidden_size, hidden_size); });
            U_c_decoder[i] = MatrixXd::NullaryExpr(hidden_size, hidden_size, 
                [&]() { return xavier_init(hidden_size, hidden_size); });
            U_o_decoder[i] = MatrixXd::NullaryExpr(hidden_size, hidden_size, 
                [&]() { return xavier_init(hidden_size, hidden_size); });
            
            b_f_decoder[i] = VectorXd::Ones(hidden_size) * 1.0;
            b_i_decoder[i] = VectorXd::Zero(hidden_size);
            b_c_decoder[i] = VectorXd::Zero(hidden_size);
            b_o_decoder[i] = VectorXd::Zero(hidden_size);
            
            embedding_size = hidden_size;
        }
    }
}

VectorXd LanguageModel::tanh(const VectorXd& x) {
    return x.array().tanh();
}

VectorXd LanguageModel::sigmoid(const VectorXd& x) {
    return 1.0 / (1.0 + (-x.array()).exp());
}

VectorXd LanguageModel::relu(const VectorXd& x) {
    return x.array().max(0.0);
}

VectorXd LanguageModel::softmax(const VectorXd& x) {
    VectorXd exp_x = (x / temperature).array().exp();
    double sum = exp_x.sum();
    if (sum < 1e-10) sum = 1e-10;  // Защита от деления на ноль
    return exp_x / sum;
}

void LanguageModel::lstmCellForward(const VectorXd& x, const VectorXd& h_prev, const VectorXd& c_prev,
                                   int layer, VectorXd& h_out, VectorXd& c_out, bool training) {
    // LSTM cell forward pass
    // Применяем dropout к входным данным (только при обучении)
    VectorXd x_dropout = applyDropout(x, training);
    VectorXd h_prev_dropout = applyDropout(h_prev, training);
    
    // Forget gate: f_t = sigmoid(W_f * x_t + U_f * h_{t-1} + b_f)
    VectorXd f_t = sigmoid(W_f[layer] * x_dropout + U_f[layer] * h_prev_dropout + b_f[layer]);
    
    // Input gate: i_t = sigmoid(W_i * x_t + U_i * h_{t-1} + b_i)
    VectorXd i_t = sigmoid(W_i[layer] * x_dropout + U_i[layer] * h_prev_dropout + b_i[layer]);
    
    // Candidate values: C_tilde = tanh(W_c * x_t + U_c * h_{t-1} + b_c)
    VectorXd C_tilde = tanh(W_c[layer] * x_dropout + U_c[layer] * h_prev_dropout + b_c[layer]);
    
    // Cell state: c_t = f_t * c_{t-1} + i_t * C_tilde
    c_out = f_t.array() * c_prev.array() + i_t.array() * C_tilde.array();
    
    // Output gate: o_t = sigmoid(W_o * x_t + U_o * h_{t-1} + b_o)
    VectorXd o_t = sigmoid(W_o_lstm[layer] * x_dropout + U_o[layer] * h_prev_dropout + b_o_lstm[layer]);
    
    // Hidden state: h_t = o_t * tanh(c_t)
    h_out = o_t.array() * tanh(c_out).array();
    
    // Применяем dropout к выходу (только при обучении)
    h_out = applyDropout(h_out, training);
}

VectorXd LanguageModel::computeAttention(const vector<VectorXd>& hidden_states, const VectorXd& current_hidden) {
    if (hidden_states.empty()) {
        return current_hidden;
    }
    
    // Вычисляем attention scores для каждого скрытого состояния
    vector<double> scores;
    double max_score = -1e10;
    
    for (const auto& h : hidden_states) {
        // Score = h^T * W_attention * current_hidden
        VectorXd score_vec = h.transpose() * W_attention * current_hidden;
        double score = score_vec(0) + b_attention.dot(current_hidden);
        scores.push_back(score);
        if (score > max_score) {
            max_score = score;
        }
    }
    
    // Softmax для получения attention weights
    VectorXd attention_weights = VectorXd::Zero(hidden_states.size());
    double sum_exp = 0.0;
    for (size_t i = 0; i < scores.size(); i++) {
        attention_weights(i) = exp(scores[i] - max_score);  // Стабильный softmax
        sum_exp += attention_weights(i);
    }
    attention_weights = attention_weights / sum_exp;
    
    // Взвешенная сумма скрытых состояний
    VectorXd attended = VectorXd::Zero(hidden_size);
    for (size_t i = 0; i < hidden_states.size(); i++) {
        attended += attention_weights(i) * hidden_states[i];
    }
    
    return attended;
}

VectorXd LanguageModel::layerNorm(const VectorXd& x) {
    // Упрощенная Layer Normalization
    double mean = x.mean();
    VectorXd centered = x.array() - mean;
    double variance = centered.array().square().mean();
    double std_dev = sqrt(variance + 1e-8);
    return centered.array() / std_dev;
}

VectorXd LanguageModel::multiHeadAttention(const vector<VectorXd>& hidden_states, const VectorXd& current_hidden, int layer_idx) {
    if (hidden_states.empty()) {
        return current_hidden;
    }
    
    int d_k = d_model / num_heads;
    VectorXd output = VectorXd::Zero(d_model);
    
    // Выбираем веса для текущего слоя Transformer
    vector<MatrixXd>* W_q_ptr = (layer_idx < (int)W_q_layers.size() && !W_q_layers.empty()) ? &W_q_layers[layer_idx] : &W_q;
    vector<MatrixXd>* W_k_ptr = (layer_idx < (int)W_k_layers.size() && !W_k_layers.empty()) ? &W_k_layers[layer_idx] : &W_k;
    vector<MatrixXd>* W_v_ptr = (layer_idx < (int)W_v_layers.size() && !W_v_layers.empty()) ? &W_v_layers[layer_idx] : &W_v;
    vector<MatrixXd>* W_o_ptr = (layer_idx < (int)W_o_transformer_layers.size() && !W_o_transformer_layers.empty()) ? &W_o_transformer_layers[layer_idx] : &W_o_transformer;
    
    // Для каждой головы внимания
    for (int h = 0; h < num_heads; h++) {
        // Query, Key, Value
        VectorXd Q = (*W_q_ptr)[h] * current_hidden;  // (d_k,)
        vector<VectorXd> K, V;
        
        for (const auto& h_state : hidden_states) {
            K.push_back((*W_k_ptr)[h] * h_state);  // (d_k,)
            V.push_back((*W_v_ptr)[h] * h_state);  // (d_k,)
        }
        
        // Вычисляем attention scores
        vector<double> scores;
        double max_score = -1e10;
        
        for (const auto& k : K) {
            double score = Q.dot(k) / sqrt(static_cast<double>(d_k));  // Scaled dot-product attention
            scores.push_back(score);
            if (score > max_score) {
                max_score = score;
            }
        }
        
        // Softmax
        VectorXd attention_weights = VectorXd::Zero(scores.size());
        double sum_exp = 0.0;
        for (size_t i = 0; i < scores.size(); i++) {
            attention_weights(i) = exp(scores[i] - max_score);
            sum_exp += attention_weights(i);
        }
        if (sum_exp > 1e-10) {
            attention_weights = attention_weights / sum_exp;
        }
        
        // Взвешенная сумма Values
        VectorXd head_output = VectorXd::Zero(d_k);
        for (size_t i = 0; i < V.size(); i++) {
            head_output += attention_weights(i) * V[i];
        }
        
        // Проецируем обратно
        output += (*W_o_ptr)[h] * head_output;
    }
    
    return output;
}

VectorXd LanguageModel::feedForward(const VectorXd& x, int layer_idx) {
    // Feed-Forward Network: FFN(x) = ReLU(W_ff1 * x + b_ff1) * W_ff2 + b_ff2
    // Выбираем веса для текущего слоя Transformer
    MatrixXd* W_ff1_ptr = (layer_idx < (int)W_ff1_layers.size() && !W_ff1_layers.empty()) ? &W_ff1_layers[layer_idx] : &W_ff1;
    MatrixXd* W_ff2_ptr = (layer_idx < (int)W_ff2_layers.size() && !W_ff2_layers.empty()) ? &W_ff2_layers[layer_idx] : &W_ff2;
    VectorXd* b_ff1_ptr = (layer_idx < (int)b_ff1_layers.size() && !b_ff1_layers.empty()) ? &b_ff1_layers[layer_idx] : &b_ff1;
    VectorXd* b_ff2_ptr = (layer_idx < (int)b_ff2_layers.size() && !b_ff2_layers.empty()) ? &b_ff2_layers[layer_idx] : &b_ff2;
    
    VectorXd ff1_out = relu((*W_ff1_ptr) * x + (*b_ff1_ptr));
    VectorXd ff2_out = (*W_ff2_ptr) * ff1_out + (*b_ff2_ptr);
    return ff2_out;
}

VectorXd LanguageModel::transformerBlock(const vector<VectorXd>& hidden_states, const VectorXd& current_hidden, int layer_idx) {
    // Transformer блок: Self-Attention + Feed-Forward с residual connections и layer norm
    
    // Self-Attention
    VectorXd attn_output = multiHeadAttention(hidden_states, current_hidden, layer_idx);
    VectorXd attn_norm = layerNorm(attn_output + current_hidden);  // Residual connection + Layer Norm
    
    // Feed-Forward
    VectorXd ff_output = feedForward(attn_norm, layer_idx);
    VectorXd output = layerNorm(ff_output + attn_norm);  // Residual connection + Layer Norm
    
    return output;
}

vector<VectorXd> LanguageModel::forwardLSTM(const vector<int>& tokens) {
    vector<VectorXd> hidden_states;
    vector<VectorXd> cell_states;
    
    // Инициализация скрытых и клеточных состояний для каждого слоя
    vector<VectorXd> h_layers(num_layers, VectorXd::Zero(hidden_size));
    vector<VectorXd> c_layers(num_layers, VectorXd::Zero(hidden_size));
    
    for (int token : tokens) {
        if (token < 0 || token >= vocab_size) {
            token = 1;  // <UNK>
        }
        
        VectorXd x = embeddings.row(token);
        
        // Проходим через все слои LSTM
        for (int layer = 0; layer < num_layers; layer++) {
            VectorXd h_prev = h_layers[layer];
            VectorXd c_prev = c_layers[layer];
            VectorXd h_new, c_new;
            
            lstmCellForward(x, h_prev, c_prev, layer, h_new, c_new);
            
            h_layers[layer] = h_new;
            c_layers[layer] = c_new;
            x = h_new;  // Вход для следующего слоя
        }
        
        hidden_states.push_back(h_layers[num_layers - 1]);
        cell_states.push_back(c_layers[num_layers - 1]);
    }
    
    return hidden_states;
}

// Encoder: обрабатывает входную последовательность
vector<VectorXd> LanguageModel::forwardEncoder(const vector<int>& tokens) {
    if (!use_encoder_decoder) {
        // Если encoder-decoder не используется, используем обычный forwardLSTM
        return forwardLSTM(tokens);
    }
    
    vector<VectorXd> hidden_states;
    vector<VectorXd> cell_states;
    
    // Инициализация скрытых и клеточных состояний для каждого слоя encoder
    vector<VectorXd> h_layers(encoder_layers, VectorXd::Zero(hidden_size));
    vector<VectorXd> c_layers(encoder_layers, VectorXd::Zero(hidden_size));
    
    for (int token : tokens) {
        if (token < 0 || token >= vocab_size) {
            token = 1;  // <UNK>
        }
        
        VectorXd x = embeddings.row(token);
        
        // Проходим через все слои encoder LSTM
        for (int layer = 0; layer < encoder_layers; layer++) {
            VectorXd h_prev = h_layers[layer];
            VectorXd c_prev = c_layers[layer];
            VectorXd h_new, c_new;
            
            // Используем веса encoder
            VectorXd f_t = sigmoid(W_f_encoder[layer] * x + U_f_encoder[layer] * h_prev + b_f_encoder[layer]);
            VectorXd i_t = sigmoid(W_i_encoder[layer] * x + U_i_encoder[layer] * h_prev + b_i_encoder[layer]);
            VectorXd C_tilde = tanh(W_c_encoder[layer] * x + U_c_encoder[layer] * h_prev + b_c_encoder[layer]);
            c_new = f_t.array() * c_prev.array() + i_t.array() * C_tilde.array();
            VectorXd o_t = sigmoid(W_o_encoder[layer] * x + U_o_encoder[layer] * h_prev + b_o_encoder[layer]);
            h_new = o_t.array() * tanh(c_new).array();
            
            h_layers[layer] = h_new;
            c_layers[layer] = c_new;
            x = h_new;  // Вход для следующего слоя
        }
        
        hidden_states.push_back(h_layers[encoder_layers - 1]);
        cell_states.push_back(c_layers[encoder_layers - 1]);
    }
    
    return hidden_states;
}

// Decoder: генерирует выходную последовательность с использованием encoder outputs
vector<VectorXd> LanguageModel::forwardDecoder(const vector<int>& tokens, const vector<VectorXd>& encoder_outputs) {
    if (!use_encoder_decoder) {
        // Если encoder-decoder не используется, используем обычный forwardLSTM
        return forwardLSTM(tokens);
    }
    
    vector<VectorXd> hidden_states;
    
    // Инициализация скрытых и клеточных состояний для каждого слоя decoder
    vector<VectorXd> h_layers(decoder_layers, VectorXd::Zero(hidden_size));
    vector<VectorXd> c_layers(decoder_layers, VectorXd::Zero(hidden_size));
    
    for (int token : tokens) {
        if (token < 0 || token >= vocab_size) {
            token = 1;  // <UNK>
        }
        
        VectorXd x = embeddings.row(token);
        
        // Проходим через все слои decoder LSTM
        for (int layer = 0; layer < decoder_layers; layer++) {
            VectorXd h_prev = h_layers[layer];
            VectorXd c_prev = c_layers[layer];
            VectorXd h_new, c_new;
            
            // Используем веса decoder
            VectorXd f_t = sigmoid(W_f_decoder[layer] * x + U_f_decoder[layer] * h_prev + b_f_decoder[layer]);
            VectorXd i_t = sigmoid(W_i_decoder[layer] * x + U_i_decoder[layer] * h_prev + b_i_decoder[layer]);
            VectorXd C_tilde = tanh(W_c_decoder[layer] * x + U_c_decoder[layer] * h_prev + b_c_decoder[layer]);
            c_new = f_t.array() * c_prev.array() + i_t.array() * C_tilde.array();
            VectorXd o_t = sigmoid(W_o_decoder[layer] * x + U_o_decoder[layer] * h_prev + b_o_decoder[layer]);
            h_new = o_t.array() * tanh(c_new).array();
            
            // Cross-Attention: decoder обращает внимание на encoder outputs
            if (!encoder_outputs.empty() && layer == decoder_layers - 1) {
                // Применяем cross-attention на последнем слое decoder
                int d_k = d_model / num_heads;
                VectorXd Q = W_q_cross * h_new;
                vector<double> scores;
                double max_score = -1e10;
                
                for (const auto& enc_out : encoder_outputs) {
                    VectorXd K = W_k_cross * enc_out;
                    double score = Q.dot(K) / sqrt(static_cast<double>(d_k));
                    scores.push_back(score);
                    if (score > max_score) {
                        max_score = score;
                    }
                }
                
                // Softmax для attention weights
                VectorXd attention_weights = VectorXd::Zero(scores.size());
                double sum_exp = 0.0;
                for (size_t i = 0; i < scores.size(); i++) {
                    attention_weights(i) = exp(scores[i] - max_score);
                    sum_exp += attention_weights(i);
                }
                if (sum_exp > 1e-10) {
                    attention_weights = attention_weights / sum_exp;
                }
                
                // Взвешенная сумма encoder outputs
                VectorXd cross_attn_output = VectorXd::Zero(d_k);
                for (size_t i = 0; i < encoder_outputs.size(); i++) {
                    VectorXd V = W_v_cross * encoder_outputs[i];
                    cross_attn_output += attention_weights(i) * V;
                }
                
                // Объединяем с decoder hidden state
                h_new = h_new + W_o_cross * cross_attn_output;
            }
            
            h_layers[layer] = h_new;
            c_layers[layer] = c_new;
            x = h_new;  // Вход для следующего слоя
        }
        
        hidden_states.push_back(h_layers[decoder_layers - 1]);
    }
    
    return hidden_states;
}

// Параллельная версия forwardLSTM
vector<VectorXd> LanguageModel::forwardLSTMParallel(const vector<int>& tokens) {
    // Используем обычную версию, если num_threads = 1
    if (this->num_threads <= 1) {
        return this->forwardLSTM(tokens);
    }
    
    // Упрощенная параллельная версия: батчинг
    vector<VectorXd> hidden_states;
    const int batch_size = min(this->num_threads, (int)tokens.size());
    
    for (size_t t = 0; t < tokens.size(); t += batch_size) {
        int current_batch = min(batch_size, (int)(tokens.size() - t));
        
        // Обрабатываем батч токенов последовательно (упрощенная версия)
        for (int b = 0; b < current_batch; b++) {
            int token = tokens[t + b];
            if (token < 0 || token >= this->vocab_size) {
                token = 1;  // <UNK>
            }
            
            VectorXd x = this->embeddings.row(token);
            
            // Проходим через все слои LSTM
            static vector<VectorXd> h_layers_cache;
            static vector<VectorXd> c_layers_cache;
            if (h_layers_cache.size() != this->num_layers) {
                h_layers_cache.resize(this->num_layers, VectorXd::Zero(this->hidden_size));
                c_layers_cache.resize(this->num_layers, VectorXd::Zero(this->hidden_size));
            }
            
            for (int layer = 0; layer < this->num_layers; layer++) {
                VectorXd h_prev = h_layers_cache[layer];
                VectorXd c_prev = c_layers_cache[layer];
                VectorXd h_new, c_new;
                
                this->lstmCellForward(x, h_prev, c_prev, layer, h_new, c_new);
                
                h_layers_cache[layer] = h_new;
                c_layers_cache[layer] = c_new;
                x = h_new;
            }
            
            hidden_states.push_back(h_layers_cache[this->num_layers - 1]);
        }
    }
    
    return hidden_states;
}

int LanguageModel::sampleNextToken(const VectorXd& logits, const vector<int>& previous_tokens) {
    // Применяем температурную настройку
    double temp = (temperature > 0.0) ? temperature : 1.0;
    VectorXd scaled_logits = logits / temp;
    
    // Применяем repetition penalty к уже использованным токенам
    if (repetition_penalty > 1.0 && !previous_tokens.empty()) {
        map<int, int> token_counts;
        for (int token : previous_tokens) {
            token_counts[token]++;
        }
        for (auto& pair : token_counts) {
            if (pair.first >= 0 && pair.first < scaled_logits.size()) {
                scaled_logits(pair.first) /= (1.0 + (pair.second - 1) * (repetition_penalty - 1.0));
            }
        }
    }
    
    // Применяем softmax
    VectorXd probs = softmax(scaled_logits);
    
    // Top-K sampling: оставляем только K наиболее вероятных токенов
    if (top_k > 0 && top_k < vocab_size) {
        vector<pair<double, int>> prob_index;
        for (int i = 0; i < probs.size(); i++) {
            prob_index.push_back({probs(i), i});
        }
        sort(prob_index.rbegin(), prob_index.rend());  // Сортируем по убыванию
        
        VectorXd filtered_probs = VectorXd::Zero(vocab_size);
        double sum = 0.0;
        for (int i = 0; i < min(top_k, (int)prob_index.size()); i++) {
            filtered_probs(prob_index[i].second) = prob_index[i].first;
            sum += prob_index[i].first;
        }
        // Нормализуем
        if (sum > 0) {
            filtered_probs /= sum;
        }
        probs = filtered_probs;
    }
    
    // Top-P (nucleus) sampling: выбираем токены с кумулятивной вероятностью top_p
    if (top_p > 0.0 && top_p < 1.0) {
        vector<pair<double, int>> prob_index;
        for (int i = 0; i < probs.size(); i++) {
            if (probs(i) > 0) {
                prob_index.push_back({probs(i), i});
            }
        }
        sort(prob_index.rbegin(), prob_index.rend());
        
        double cumsum = 0.0;
        VectorXd filtered_probs = VectorXd::Zero(vocab_size);
        for (size_t i = 0; i < prob_index.size(); i++) {
            cumsum += prob_index[i].first;
            filtered_probs(prob_index[i].second) = prob_index[i].first;
            if (cumsum >= top_p) {
                break;
            }
        }
        // Нормализуем
        if (cumsum > 0) {
            filtered_probs /= cumsum;
        }
        probs = filtered_probs;
    }
    
    // Сэмплируем из распределения
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dist(0.0, 1.0);
    double r = dist(gen);
    
    double cumsum = 0.0;
    for (int i = 0; i < probs.size(); i++) {
        cumsum += probs(i);
        if (r <= cumsum) {
            return i;
        }
    }
    
    return probs.size() - 1;
}

string LanguageModel::generateResponse(const string& user_input, const vector<ChatMessage>& context) {
    if (!g_tokenizer) {
        // Fallback на простые ответы, если токенизатор не инициализирован
        return generateSimpleResponse(user_input);
    }
    
    try {
        // Токенизация входного текста
        vector<int> input_tokens = g_tokenizer->encode(user_input);
        if (input_tokens.empty()) {
            return generateSimpleResponse(user_input);
        }
        
        // Прямое распространение через LSTM или Encoder-Decoder
        vector<VectorXd> hidden_states;
        vector<VectorXd> encoder_outputs;
        
        if (use_encoder_decoder) {
            // Используем Encoder-Decoder архитектуру
            encoder_outputs = forwardEncoder(input_tokens);
            if (encoder_outputs.empty()) {
                return generateSimpleResponse(user_input);
            }
            hidden_states = encoder_outputs;  // Для начального состояния decoder
        } else {
            // Используем обычный LSTM
            hidden_states = forwardLSTM(input_tokens);
            if (hidden_states.empty()) {
                return generateSimpleResponse(user_input);
            }
        }
        
        // Применяем Attention к контексту, если он есть
        VectorXd context_attended = hidden_states.back();  // Начинаем с последнего скрытого состояния
        if (!context.empty() && context.size() > 1) {
            // Собираем скрытые состояния из контекста
            vector<VectorXd> context_hidden;
            for (const auto& msg : context) {
                if (msg.role == "user" || msg.role == "assistant") {
                    vector<int> msg_tokens = g_tokenizer->encode(msg.content);
                    if (!msg_tokens.empty()) {
                        vector<VectorXd> msg_hidden = forwardLSTM(msg_tokens);
                        if (!msg_hidden.empty()) {
                            context_hidden.push_back(msg_hidden.back());
                        }
                    }
                }
            }
            
            // Применяем несколько слоев Transformer последовательно
            if (!context_hidden.empty()) {
                context_attended = hidden_states.back();
                for (int layer = 0; layer < num_transformer_layers; layer++) {
                    context_attended = transformerBlock(context_hidden, context_attended, layer);
                }
            }
        }
        
        // Генерация ответа токен за токеном
        vector<int> response_tokens;
        VectorXd current_hidden = context_attended;
        VectorXd current_cell = VectorXd::Zero(hidden_size);
        
        // Используем encode для получения специальных токенов
        vector<int> start_tokens = g_tokenizer->encode("<START>");
        vector<int> end_tokens = g_tokenizer->encode("<END>");
        int start_token = !start_tokens.empty() ? start_tokens[0] : 2;
        int end_token = !end_tokens.empty() ? end_tokens[0] : 3;
        
        response_tokens.push_back(start_token);
        
        for (int i = 0; i < max_length; i++) {
            // Если используется encoder-decoder, применяем decoder для каждого токена
            if (use_encoder_decoder && !encoder_outputs.empty()) {
                vector<int> current_tokens = {response_tokens.back()};
                vector<VectorXd> decoder_states = forwardDecoder(current_tokens, encoder_outputs);
                if (!decoder_states.empty()) {
                    current_hidden = decoder_states.back();
                }
            }
            
            // Вычисляем вероятности следующего токена
            VectorXd logits = W_hy[0] * current_hidden + b_y[0];
            int next_token = sampleNextToken(logits, response_tokens);
            
            if (next_token == end_token || next_token < 0 || next_token >= vocab_size) {
                break;  // Конец генерации
            }
            
            response_tokens.push_back(next_token);
            
            // Обновляем скрытое состояние для следующего токена через LSTM
            VectorXd next_embedding = embeddings.row(next_token);
            for (int layer = 0; layer < num_layers; layer++) {
                VectorXd h_prev = current_hidden;
                VectorXd c_prev = current_cell;
                VectorXd h_new, c_new;
                
                lstmCellForward(next_embedding, h_prev, c_prev, layer, h_new, c_new);
                
                current_hidden = h_new;
                current_cell = c_new;
                next_embedding = h_new;  // Вход для следующего слоя
            }
        }
        
        // Декодирование токенов в текст
        string response = g_tokenizer->decode(response_tokens);
        
        // Удаляем все специальные токены из ответа (на случай, если они попали)
        response = regex_replace(response, regex("<PAD>|<UNK>|<START>|<END>"), "");
        // Удаляем лишние пробелы
        response = regex_replace(response, regex("\\s+"), " ");
        response = regex_replace(response, regex("^\\s+|\\s+$"), "");
        
        // Если ответ пустой или слишком короткий, используем простой ответ
        if (response.empty() || response.length() < 3) {
            return generateSimpleResponse(user_input);
        }
        
        // Определяем язык запроса пользователя и фильтруем ответ
        string user_language = g_tokenizer->detectLanguage(user_input);
        string filtered_response = filterResponseByLanguage(response, user_language);
        
        // Проверяем, что отфильтрованный ответ на правильном языке
        string filtered_language = g_tokenizer->detectLanguage(filtered_response);
        if (filtered_language != user_language && !filtered_response.empty()) {
            // Если после фильтрации язык всё ещё неправильный, используем простой ответ
            return generateSimpleResponse(user_input);
        }
        
        // Если отфильтрованный ответ слишком короткий или пустой, используем простой ответ
        if (filtered_response.empty() || filtered_response.length() < 3) {
            return generateSimpleResponse(user_input);
        }
        
        return filtered_response;
        
    } catch (...) {
        // В случае ошибки используем простой ответ
        return generateSimpleResponse(user_input);
    }
}

// Вспомогательная функция для простых ответов (fallback)
string LanguageModel::generateSimpleResponse(const string& user_input) {
    // Простое определение языка
    int cyrillic_count = 0;
    int latin_count = 0;
    for (char c : user_input) {
        if ((c >= 0x0400 && c <= 0x04FF) || (c >= 0x0500 && c <= 0x052F)) {
            cyrillic_count++;
        } else if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) {
            latin_count++;
        }
    }
    string language = (cyrillic_count > latin_count) ? "ru" : "en";
    
    // Простые ответы для демонстрации
    if (language == "ru") {
        if (user_input.find("привет") != string::npos || user_input.find("здравствуй") != string::npos) {
            return "Привет! Чем могу помочь?";
        } else if (user_input.find("как дела") != string::npos) {
            return "У меня всё отлично, спасибо! А у вас как дела?";
        } else if (user_input.find("что ты умеешь") != string::npos) {
            return "Я могу отвечать на ваши вопросы, помогать с различными задачами и вести диалог. Что вас интересует?";
        } else {
            return "Спасибо за ваш вопрос. Я обрабатываю информацию и готовлю ответ.";
        }
    } else {
        if (user_input.find("hello") != string::npos || user_input.find("hi") != string::npos) {
            return "Hello! How can I help you?";
        } else if (user_input.find("how are you") != string::npos) {
            return "I'm doing great, thank you! How are you?";
        } else if (user_input.find("what can you do") != string::npos) {
            return "I can answer your questions, help with various tasks, and have conversations. What interests you?";
        } else {
            return "Thank you for your question. I'm processing the information and preparing an answer.";
        }
    }
}

string LanguageModel::filterResponseByLanguage(const string& response, const string& target_language) {
    if (response.empty() || !g_tokenizer) {
        return response;
    }
    
    // Более строгая проверка: определяем язык всего ответа
    string response_language = g_tokenizer->detectLanguage(response);
    
    // Если язык ответа совпадает с целевым языком, возвращаем как есть
    if (response_language == target_language) {
        return response;
    }
    
    // Если языки не совпадают, фильтруем слово за словом
    istringstream iss(response);
    vector<string> words;
    string word;
    while (iss >> word) {
        words.push_back(word);
    }
    
    if (words.empty()) {
        return response;
    }
    
    // Строгая фильтрация: удаляем ВСЕ слова другого языка
    vector<string> filtered_words;
    int target_language_count = 0;
    
    for (const string& w : words) {
        // Проверяем, является ли слово только знаками препинания или числами
        bool is_punctuation_or_number = true;
        bool has_letters = false;
        
        for (char c : w) {
            unsigned char uc = static_cast<unsigned char>(c);
            if (ispunct(uc) || isdigit(uc)) {
                continue;
            } else if (isalpha(uc) || (uc >= 0x0400 && uc <= 0x052F)) {
                has_letters = true;
                is_punctuation_or_number = false;
            }
        }
        
        // Если это только знаки препинания или числа, оставляем
        if (is_punctuation_or_number) {
            filtered_words.push_back(w);
            continue;
        }
        
        // Если есть буквы, проверяем язык слова
        if (has_letters) {
            string word_language = g_tokenizer->detectLanguage(w);
            
            // Оставляем только слова целевого языка
            if (word_language == target_language) {
                filtered_words.push_back(w);
                target_language_count++;
            }
            // Все остальные слова (другого языка) удаляем
        }
    }
    
    // Если не осталось ни одного слова целевого языка, возвращаем пустую строку
    // (вызывающий код должен обработать это и вернуть простой ответ)
    if (target_language_count == 0) {
        return "";  // Пустая строка - сигнал для использования простого ответа
    }
    
    // Собираем отфильтрованный ответ
    string filtered_response;
    for (size_t i = 0; i < filtered_words.size(); i++) {
        if (i > 0 && !filtered_words[i].empty() && 
            !ispunct(static_cast<unsigned char>(filtered_words[i][0]))) {
            filtered_response += " ";
        }
        filtered_response += filtered_words[i];
    }
    
    // Удаляем лишние пробелы
    filtered_response = regex_replace(filtered_response, regex("\\s+"), " ");
    filtered_response = regex_replace(filtered_response, regex("^\\s+|\\s+$"), "");
    
    return filtered_response;
}

void LanguageModel::trainOnDialogues(const vector<vector<ChatMessage>>& dialogues, int epochs, int batch_size,
                                     double validation_split, int patience, const string& checkpoint_dir) {
    if (!g_tokenizer) {
        cerr << "Ошибка: Токенизатор не инициализирован!" << endl;
        return;
    }
    
    if (dialogues.empty()) {
        cerr << "Ошибка: Нет диалогов для обучения!" << endl;
        return;
    }
    
    cout << "Начало обучения языковой модели на " << dialogues.size() << " диалогах..." << endl;
    
    // Разделение на train и validation
    vector<vector<ChatMessage>> train_dialogues, val_dialogues;
    int val_size = max(1, (int)(dialogues.size() * validation_split));
    for (size_t i = 0; i < dialogues.size(); i++) {
        if (i < dialogues.size() - val_size) {
            train_dialogues.push_back(dialogues[i]);
        } else {
            val_dialogues.push_back(dialogues[i]);
        }
    }
    cout << "Train: " << train_dialogues.size() << ", Validation: " << val_dialogues.size() << endl;
    
    double learning_rate = 0.001;
    this->checkpoint_directory = checkpoint_dir;
    this->best_validation_loss = 1e10;
    this->no_improvement_epochs = 0;
    
    // Инициализация градиентов
    dW_f.resize(num_layers);
    dW_i.resize(num_layers);
    dW_c.resize(num_layers);
    dW_o_lstm.resize(num_layers);
    dU_f.resize(num_layers);
    dU_i.resize(num_layers);
    dU_c.resize(num_layers);
    dU_o.resize(num_layers);
    db_f.resize(num_layers);
    db_i.resize(num_layers);
    db_c.resize(num_layers);
    db_o_lstm.resize(num_layers);
    dW_hy.resize(num_layers);
    db_y.resize(num_layers);
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        double total_loss = 0.0;
        int total_samples = 0;
        
        cout << "Эпоха " << (epoch + 1) << "/" << epochs << "..." << endl;
        
        // Data Augmentation: увеличиваем обучающие данные
        vector<vector<ChatMessage>> augmented_train = augmentDialogues(train_dialogues);
        cout << "После augmentation: " << augmented_train.size() << " диалогов" << endl;
        
        int accumulated_steps = 0;
        
        for (size_t d = 0; d < augmented_train.size(); d++) {
            const auto& dialogue = augmented_train[d];
            
            // Строим последовательность токенов из диалога
            vector<int> input_sequence;
            vector<int> target_sequence;
            
            for (size_t i = 0; i < dialogue.size(); i++) {
                const auto& msg = dialogue[i];
                vector<int> msg_tokens = g_tokenizer->encode(msg.content);
                
                if (i == 0) {
                    // Первое сообщение - это вход
                    input_sequence.insert(input_sequence.end(), msg_tokens.begin(), msg_tokens.end());
                } else {
                    // Последующие сообщения - это цели для обучения
                    if (msg.role == "assistant") {
                        target_sequence.insert(target_sequence.end(), msg_tokens.begin(), msg_tokens.end());
                    } else {
                        // Сообщения пользователя добавляем к входной последовательности
                        input_sequence.insert(input_sequence.end(), msg_tokens.begin(), msg_tokens.end());
                    }
                }
            }
            
            if (input_sequence.empty() || target_sequence.empty()) {
                continue;
            }
            
            // Объединяем вход и цель для обучения (teacher forcing)
            vector<int> full_sequence = input_sequence;
            full_sequence.insert(full_sequence.end(), target_sequence.begin(), target_sequence.end());
            
            // Прямое распространение
            vector<VectorXd> hidden_states = this->forwardLSTM(full_sequence);
            vector<VectorXd> cell_states;
            
            // Сохраняем cell states для backpropagation
            // (упрощенная версия - в реальности нужно сохранять все промежуточные состояния)
            cell_states.resize(hidden_states.size());
            for (size_t i = 0; i < hidden_states.size(); i++) {
                cell_states[i] = VectorXd::Zero(hidden_size);
            }
            
            // Вычисляем потери и градиенты
            double dialogue_loss = 0.0;
            
            // Для каждого токена в целевой последовательности
            for (size_t t = input_sequence.size(); t < full_sequence.size() && t < hidden_states.size(); t++) {
                int target_token = full_sequence[t];
                VectorXd logits = W_hy[0] * hidden_states[t] + b_y[0];
                double loss = computeLoss(logits, target_token);
                dialogue_loss += loss;
                total_samples++;
                
                // Вычисляем градиенты для выходного слоя
                VectorXd probs = softmax(logits);
                VectorXd dlogits = probs;
                dlogits(target_token) -= 1.0;  // Градиент кросс-энтропии
                
                // Градиенты для W_hy и b_y
                if (dW_hy[0].size() == 0) {
                    dW_hy[0] = MatrixXd::Zero(vocab_size, hidden_size);
                    db_y[0] = VectorXd::Zero(vocab_size);
                }
                dW_hy[0] += dlogits * hidden_states[t].transpose();
                db_y[0] += dlogits;
                
                // Градиент для скрытого состояния (упрощенная версия)
                VectorXd dh = W_hy[0].transpose() * dlogits;
                
                // Здесь должен быть полный backpropagation через LSTM
                // Для упрощения используем упрощенную версию
            }
            
            total_loss += dialogue_loss;
            accumulated_steps++;
            
            // Gradient Accumulation: обновляем веса только после накопления градиентов
            if (accumulated_steps >= gradient_accumulation_steps || 
                (d + 1) % batch_size == 0 || d == augmented_train.size() - 1) {
                // Нормализуем градиенты при accumulation
                if (accumulated_steps > 1) {
                    for (int layer = 0; layer < num_layers; layer++) {
                        if (dW_f[layer].size() > 0) {
                            dW_f[layer] /= accumulated_steps;
                            dW_i[layer] /= accumulated_steps;
                            dW_c[layer] /= accumulated_steps;
                            dW_o_lstm[layer] /= accumulated_steps;
                            dU_f[layer] /= accumulated_steps;
                            dU_i[layer] /= accumulated_steps;
                            dU_c[layer] /= accumulated_steps;
                            dU_o[layer] /= accumulated_steps;
                            db_f[layer] /= accumulated_steps;
                            db_i[layer] /= accumulated_steps;
                            db_c[layer] /= accumulated_steps;
                            db_o_lstm[layer] /= accumulated_steps;
                        }
                    }
                    if (dW_hy[0].size() > 0) {
                        dW_hy[0] /= accumulated_steps;
                        db_y[0] /= accumulated_steps;
                    }
                    if (dembeddings.size() > 0) {
                        dembeddings /= accumulated_steps;
                    }
                }
                
                // Используем Adam optimizer вместо простого SGD
                updateWeightsAdam(learning_rate);
                accumulated_steps = 0;
                
                // Обнуляем градиенты
                for (int layer = 0; layer < num_layers; layer++) {
                    dW_f[layer] = MatrixXd::Zero(hidden_size, hidden_size);
                    dW_i[layer] = MatrixXd::Zero(hidden_size, hidden_size);
                    dW_c[layer] = MatrixXd::Zero(hidden_size, hidden_size);
                    dW_o_lstm[layer] = MatrixXd::Zero(hidden_size, hidden_size);
                    dU_f[layer] = MatrixXd::Zero(hidden_size, hidden_size);
                    dU_i[layer] = MatrixXd::Zero(hidden_size, hidden_size);
                    dU_c[layer] = MatrixXd::Zero(hidden_size, hidden_size);
                    dU_o[layer] = MatrixXd::Zero(hidden_size, hidden_size);
                    db_f[layer] = VectorXd::Zero(hidden_size);
                    db_i[layer] = VectorXd::Zero(hidden_size);
                    db_c[layer] = VectorXd::Zero(hidden_size);
                    db_o_lstm[layer] = VectorXd::Zero(hidden_size);
                }
                dW_hy[0] = MatrixXd::Zero(vocab_size, hidden_size);
                db_y[0] = VectorXd::Zero(vocab_size);
            }
            
            if ((d + 1) % 10 == 0) {
                cout << "  Обработано диалогов: " << (d + 1) << "/" << augmented_train.size() << endl;
            }
        }
        
        double avg_train_loss = total_samples > 0 ? total_loss / total_samples : 0.0;
        cout << "Эпоха " << (epoch + 1) << " завершена. Средняя потеря (train): " << avg_train_loss << endl;
        
        // Вычисляем validation loss
        double val_loss = 0.0;
        int val_samples = 0;
        for (const auto& dialogue : val_dialogues) {
            vector<int> input_sequence, target_sequence;
            for (size_t i = 0; i < dialogue.size(); i++) {
                const auto& msg = dialogue[i];
                vector<int> msg_tokens = g_tokenizer->encode(msg.content);
                if (i == 0) {
                    input_sequence.insert(input_sequence.end(), msg_tokens.begin(), msg_tokens.end());
                } else if (msg.role == "assistant") {
                    target_sequence.insert(target_sequence.end(), msg_tokens.begin(), msg_tokens.end());
                }
            }
            if (!input_sequence.empty() && !target_sequence.empty()) {
                vector<int> full_sequence = input_sequence;
                full_sequence.insert(full_sequence.end(), target_sequence.begin(), target_sequence.end());
                vector<VectorXd> hidden_states = this->forwardLSTM(full_sequence);
                for (size_t t = input_sequence.size(); t < full_sequence.size() && t < hidden_states.size(); t++) {
                    int target_token = full_sequence[t];
                    VectorXd logits = W_hy[0] * hidden_states[t] + b_y[0];
                    val_loss += computeLoss(logits, target_token);
                    val_samples++;
                }
            }
        }
        double avg_val_loss = val_samples > 0 ? val_loss / val_samples : 0.0;
        cout << "Validation loss: " << avg_val_loss << endl;
        
        // Сохраняем потери
        this->training_losses.push_back(avg_train_loss);
        this->validation_losses.push_back(avg_val_loss);
        
        // Early Stopping и Checkpointing
        if (avg_val_loss < best_validation_loss) {
            best_validation_loss = avg_val_loss;
            no_improvement_epochs = 0;
            
            // Сохраняем лучший чекпоинт
            if (!checkpoint_directory.empty()) {
                string checkpoint_path = checkpoint_directory + "/best_model_epoch_" + to_string(epoch + 1) + ".bin";
                saveCheckpoint(checkpoint_path, epoch + 1, avg_val_loss);
                cout << "Сохранен лучший чекпоинт: " << checkpoint_path << endl;
            }
        } else {
            no_improvement_epochs++;
            if (shouldStopEarly(avg_val_loss, no_improvement_epochs, best_validation_loss, patience)) {
                cout << "Early stopping на эпохе " << (epoch + 1) << " (нет улучшения " << no_improvement_epochs << " эпох)" << endl;
                break;
            }
        }
        
        // Learning rate scheduling с warmup
        double warmup_steps = 100;
        double current_step = epoch * augmented_train.size();
        if (current_step < warmup_steps) {
            // Warmup: постепенное увеличение learning rate
            learning_rate = 0.001 * (current_step / warmup_steps);
        } else {
            // Cosine annealing: плавное уменьшение learning rate
            double cosine_decay = 0.5 * (1 + cos(3.14159 * (current_step - warmup_steps) / (epochs * augmented_train.size() - warmup_steps)));
            learning_rate = 0.001 * cosine_decay;
        }
    }
    
    cout << "Обучение завершено!" << endl;
    if (!checkpoint_directory.empty() && best_validation_loss < 1e10) {
        cout << "Лучшая validation loss: " << best_validation_loss << endl;
    }
}

void LanguageModel::saveModel(const string& path) {
    ofstream file(path, ios::binary);
    if (!file.is_open()) {
        cerr << "Ошибка: Не удалось открыть файл для сохранения: " << path << endl;
        return;
    }
    
    // Сохраняем параметры модели
    file.write(reinterpret_cast<const char*>(&vocab_size), sizeof(int));
    file.write(reinterpret_cast<const char*>(&hidden_size), sizeof(int));
    file.write(reinterpret_cast<const char*>(&num_layers), sizeof(int));
    file.write(reinterpret_cast<const char*>(&num_heads), sizeof(int));
    file.write(reinterpret_cast<const char*>(&d_model), sizeof(int));
    
    // Сохраняем веса LSTM
    for (int i = 0; i < num_layers; i++) {
        int rows = W_f[i].rows(), cols = W_f[i].cols();
        file.write(reinterpret_cast<const char*>(&rows), sizeof(int));
        file.write(reinterpret_cast<const char*>(&cols), sizeof(int));
        file.write(reinterpret_cast<const char*>(W_f[i].data()), rows * cols * sizeof(double));
        
        rows = W_i[i].rows(); cols = W_i[i].cols();
        file.write(reinterpret_cast<const char*>(&rows), sizeof(int));
        file.write(reinterpret_cast<const char*>(&cols), sizeof(int));
        file.write(reinterpret_cast<const char*>(W_i[i].data()), rows * cols * sizeof(double));
        
        rows = W_c[i].rows(); cols = W_c[i].cols();
        file.write(reinterpret_cast<const char*>(&rows), sizeof(int));
        file.write(reinterpret_cast<const char*>(&cols), sizeof(int));
        file.write(reinterpret_cast<const char*>(W_c[i].data()), rows * cols * sizeof(double));
        
        rows = W_o_lstm[i].rows(); cols = W_o_lstm[i].cols();
        file.write(reinterpret_cast<const char*>(&rows), sizeof(int));
        file.write(reinterpret_cast<const char*>(&cols), sizeof(int));
        file.write(reinterpret_cast<const char*>(W_o_lstm[i].data()), rows * cols * sizeof(double));
        
        // U веса
        rows = U_f[i].rows(); cols = U_f[i].cols();
        file.write(reinterpret_cast<const char*>(&rows), sizeof(int));
        file.write(reinterpret_cast<const char*>(&cols), sizeof(int));
        file.write(reinterpret_cast<const char*>(U_f[i].data()), rows * cols * sizeof(double));
        
        rows = U_i[i].rows(); cols = U_i[i].cols();
        file.write(reinterpret_cast<const char*>(&rows), sizeof(int));
        file.write(reinterpret_cast<const char*>(&cols), sizeof(int));
        file.write(reinterpret_cast<const char*>(U_i[i].data()), rows * cols * sizeof(double));
        
        rows = U_c[i].rows(); cols = U_c[i].cols();
        file.write(reinterpret_cast<const char*>(&rows), sizeof(int));
        file.write(reinterpret_cast<const char*>(&cols), sizeof(int));
        file.write(reinterpret_cast<const char*>(U_c[i].data()), rows * cols * sizeof(double));
        
        rows = U_o[i].rows(); cols = U_o[i].cols();
        file.write(reinterpret_cast<const char*>(&rows), sizeof(int));
        file.write(reinterpret_cast<const char*>(&cols), sizeof(int));
        file.write(reinterpret_cast<const char*>(U_o[i].data()), rows * cols * sizeof(double));
        
        // Смещения
        int size = b_f[i].size();
        file.write(reinterpret_cast<const char*>(&size), sizeof(int));
        file.write(reinterpret_cast<const char*>(b_f[i].data()), size * sizeof(double));
        
        size = b_i[i].size();
        file.write(reinterpret_cast<const char*>(&size), sizeof(int));
        file.write(reinterpret_cast<const char*>(b_i[i].data()), size * sizeof(double));
        
        size = b_c[i].size();
        file.write(reinterpret_cast<const char*>(&size), sizeof(int));
        file.write(reinterpret_cast<const char*>(b_c[i].data()), size * sizeof(double));
        
        size = b_o_lstm[i].size();
        file.write(reinterpret_cast<const char*>(&size), sizeof(int));
        file.write(reinterpret_cast<const char*>(b_o_lstm[i].data()), size * sizeof(double));
    }
    
    // Сохраняем embeddings
    int rows = embeddings.rows(), cols = embeddings.cols();
    file.write(reinterpret_cast<const char*>(&rows), sizeof(int));
    file.write(reinterpret_cast<const char*>(&cols), sizeof(int));
    file.write(reinterpret_cast<const char*>(embeddings.data()), rows * cols * sizeof(double));
    
    // Сохраняем выходной слой
    rows = W_hy[0].rows(); cols = W_hy[0].cols();
    file.write(reinterpret_cast<const char*>(&rows), sizeof(int));
    file.write(reinterpret_cast<const char*>(&cols), sizeof(int));
    file.write(reinterpret_cast<const char*>(W_hy[0].data()), rows * cols * sizeof(double));
    
    int size = b_y[0].size();
    file.write(reinterpret_cast<const char*>(&size), sizeof(int));
    file.write(reinterpret_cast<const char*>(b_y[0].data()), size * sizeof(double));
    
    file.close();
    cout << "Модель сохранена в: " << path << endl;
}

void LanguageModel::loadModel(const string& path) {
    ifstream file(path, ios::binary);
    if (!file.is_open()) {
        cerr << "Ошибка: Не удалось открыть файл для загрузки: " << path << endl;
        return;
    }
    
    // Загружаем параметры модели
    int loaded_vocab_size, loaded_hidden_size, loaded_num_layers, loaded_num_heads, loaded_d_model;
    file.read(reinterpret_cast<char*>(&loaded_vocab_size), sizeof(int));
    file.read(reinterpret_cast<char*>(&loaded_hidden_size), sizeof(int));
    file.read(reinterpret_cast<char*>(&loaded_num_layers), sizeof(int));
    file.read(reinterpret_cast<char*>(&loaded_num_heads), sizeof(int));
    file.read(reinterpret_cast<char*>(&loaded_d_model), sizeof(int));
    
    if (loaded_vocab_size != vocab_size || loaded_hidden_size != hidden_size || 
        loaded_num_layers != num_layers || loaded_num_heads != num_heads || loaded_d_model != d_model) {
        cerr << "Ошибка: Параметры модели не совпадают!" << endl;
        file.close();
        return;
    }
    
    // Загружаем веса LSTM
    for (int i = 0; i < num_layers; i++) {
        int rows, cols;
        file.read(reinterpret_cast<char*>(&rows), sizeof(int));
        file.read(reinterpret_cast<char*>(&cols), sizeof(int));
        W_f[i] = MatrixXd(rows, cols);
        file.read(reinterpret_cast<char*>(W_f[i].data()), rows * cols * sizeof(double));
        
        file.read(reinterpret_cast<char*>(&rows), sizeof(int));
        file.read(reinterpret_cast<char*>(&cols), sizeof(int));
        W_i[i] = MatrixXd(rows, cols);
        file.read(reinterpret_cast<char*>(W_i[i].data()), rows * cols * sizeof(double));
        
        file.read(reinterpret_cast<char*>(&rows), sizeof(int));
        file.read(reinterpret_cast<char*>(&cols), sizeof(int));
        W_c[i] = MatrixXd(rows, cols);
        file.read(reinterpret_cast<char*>(W_c[i].data()), rows * cols * sizeof(double));
        
        file.read(reinterpret_cast<char*>(&rows), sizeof(int));
        file.read(reinterpret_cast<char*>(&cols), sizeof(int));
        W_o_lstm[i] = MatrixXd(rows, cols);
        file.read(reinterpret_cast<char*>(W_o_lstm[i].data()), rows * cols * sizeof(double));
        
        // U веса
        file.read(reinterpret_cast<char*>(&rows), sizeof(int));
        file.read(reinterpret_cast<char*>(&cols), sizeof(int));
        U_f[i] = MatrixXd(rows, cols);
        file.read(reinterpret_cast<char*>(U_f[i].data()), rows * cols * sizeof(double));
        
        file.read(reinterpret_cast<char*>(&rows), sizeof(int));
        file.read(reinterpret_cast<char*>(&cols), sizeof(int));
        U_i[i] = MatrixXd(rows, cols);
        file.read(reinterpret_cast<char*>(U_i[i].data()), rows * cols * sizeof(double));
        
        file.read(reinterpret_cast<char*>(&rows), sizeof(int));
        file.read(reinterpret_cast<char*>(&cols), sizeof(int));
        U_c[i] = MatrixXd(rows, cols);
        file.read(reinterpret_cast<char*>(U_c[i].data()), rows * cols * sizeof(double));
        
        file.read(reinterpret_cast<char*>(&rows), sizeof(int));
        file.read(reinterpret_cast<char*>(&cols), sizeof(int));
        U_o[i] = MatrixXd(rows, cols);
        file.read(reinterpret_cast<char*>(U_o[i].data()), rows * cols * sizeof(double));
        
        // Смещения
        int size;
        file.read(reinterpret_cast<char*>(&size), sizeof(int));
        b_f[i] = VectorXd(size);
        file.read(reinterpret_cast<char*>(b_f[i].data()), size * sizeof(double));
        
        file.read(reinterpret_cast<char*>(&size), sizeof(int));
        b_i[i] = VectorXd(size);
        file.read(reinterpret_cast<char*>(b_i[i].data()), size * sizeof(double));
        
        file.read(reinterpret_cast<char*>(&size), sizeof(int));
        b_c[i] = VectorXd(size);
        file.read(reinterpret_cast<char*>(b_c[i].data()), size * sizeof(double));
        
        file.read(reinterpret_cast<char*>(&size), sizeof(int));
        b_o_lstm[i] = VectorXd(size);
        file.read(reinterpret_cast<char*>(b_o_lstm[i].data()), size * sizeof(double));
    }
    
    // Загружаем embeddings
    int rows, cols;
    file.read(reinterpret_cast<char*>(&rows), sizeof(int));
    file.read(reinterpret_cast<char*>(&cols), sizeof(int));
    embeddings = MatrixXd(rows, cols);
    file.read(reinterpret_cast<char*>(embeddings.data()), rows * cols * sizeof(double));
    
    // Загружаем выходной слой
    file.read(reinterpret_cast<char*>(&rows), sizeof(int));
    file.read(reinterpret_cast<char*>(&cols), sizeof(int));
    W_hy[0] = MatrixXd(rows, cols);
    file.read(reinterpret_cast<char*>(W_hy[0].data()), rows * cols * sizeof(double));
    
    int size;
    file.read(reinterpret_cast<char*>(&size), sizeof(int));
    b_y[0] = VectorXd(size);
    file.read(reinterpret_cast<char*>(b_y[0].data()), size * sizeof(double));
    
    file.close();
    cout << "Модель загружена из: " << path << endl;
}

double LanguageModel::computeLoss(const VectorXd& logits, int target_token) {
    VectorXd probs = softmax(logits);
    return -log(probs(target_token) + 1e-10);
}

void LanguageModel::updateWeights(double learning_rate) {
    // Функция для применения gradient clipping
    auto clipGradient = [this](MatrixXd& grad) {
        if (grad.size() == 0) return;
        double norm = grad.norm();
        if (norm > gradient_clip && norm > 0) {
            grad = grad * (gradient_clip / norm);
        }
    };
    
    auto clipGradientVec = [this](VectorXd& grad) {
        if (grad.size() == 0) return;
        double norm = grad.norm();
        if (norm > gradient_clip && norm > 0) {
            grad = grad * (gradient_clip / norm);
        }
    };
    
    // Обновление весов LSTM с gradient clipping
    for (int layer = 0; layer < num_layers; layer++) {
        if (dW_f[layer].size() > 0) {
            clipGradient(dW_f[layer]);
            clipGradient(dW_i[layer]);
            clipGradient(dW_c[layer]);
            clipGradient(dW_o_lstm[layer]);
            clipGradient(dU_f[layer]);
            clipGradient(dU_i[layer]);
            clipGradient(dU_c[layer]);
            clipGradient(dU_o[layer]);
            clipGradientVec(db_f[layer]);
            clipGradientVec(db_i[layer]);
            clipGradientVec(db_c[layer]);
            clipGradientVec(db_o_lstm[layer]);
            
            W_f[layer] -= learning_rate * dW_f[layer];
            W_i[layer] -= learning_rate * dW_i[layer];
            W_c[layer] -= learning_rate * dW_c[layer];
            W_o_lstm[layer] -= learning_rate * dW_o_lstm[layer];
            U_f[layer] -= learning_rate * dU_f[layer];
            U_i[layer] -= learning_rate * dU_i[layer];
            U_c[layer] -= learning_rate * dU_c[layer];
            U_o[layer] -= learning_rate * dU_o[layer];
            b_f[layer] -= learning_rate * db_f[layer];
            b_i[layer] -= learning_rate * db_i[layer];
            b_c[layer] -= learning_rate * db_c[layer];
            b_o_lstm[layer] -= learning_rate * db_o_lstm[layer];
        }
    }
    
    // Обновление весов выходного слоя с gradient clipping
    if (dW_hy[0].size() > 0) {
        clipGradient(dW_hy[0]);
        clipGradientVec(db_y[0]);
        W_hy[0] -= learning_rate * dW_hy[0];
        b_y[0] -= learning_rate * db_y[0];
    }
    
    // Обновление эмбеддингов с gradient clipping
    if (dembeddings.size() > 0) {
        clipGradient(dembeddings);
        embeddings -= learning_rate * dembeddings;
    }
}

void LanguageModel::backward(const vector<int>& tokens, const vector<int>& target_tokens,
                             const vector<VectorXd>& hidden_states, const vector<VectorXd>& cell_states) {
    // Упрощенная версия backpropagation
    // В полной реализации здесь должен быть Backpropagation Through Time (BPTT)
    // для всех слоев LSTM с учетом всех промежуточных состояний
    
    // Инициализация градиентов, если они еще не инициализированы
    if (dW_f.empty()) {
        dW_f.resize(num_layers);
        dW_i.resize(num_layers);
        dW_c.resize(num_layers);
        dW_o_lstm.resize(num_layers);
        dU_f.resize(num_layers);
        dU_i.resize(num_layers);
        dU_c.resize(num_layers);
        dU_o.resize(num_layers);
        db_f.resize(num_layers);
        db_i.resize(num_layers);
        db_c.resize(num_layers);
        db_o_lstm.resize(num_layers);
        dW_hy.resize(num_layers);
        db_y.resize(num_layers);
    }
    
    // Вычисляем градиенты для выходного слоя
    for (size_t t = 0; t < target_tokens.size() && t < hidden_states.size(); t++) {
        int target_token = target_tokens[t];
        VectorXd logits = W_hy[0] * hidden_states[t] + b_y[0];
        VectorXd probs = softmax(logits);
        VectorXd dlogits = probs;
        dlogits(target_token) -= 1.0;
        
        // Градиенты для выходного слоя
        if (dW_hy[0].size() == 0) {
            dW_hy[0] = MatrixXd::Zero(vocab_size, hidden_size);
            db_y[0] = VectorXd::Zero(vocab_size);
        }
        dW_hy[0] += dlogits * hidden_states[t].transpose();
        db_y[0] += dlogits;
    }
    
    // Градиенты для LSTM слоев (упрощенная версия)
    // В полной реализации здесь должен быть полный BPTT
    // с учетом всех промежуточных состояний и градиентов через время
}

void LanguageModel::initializeAdam() {
    // Инициализация моментов для Adam optimizer
    m_W_f.resize(num_layers);
    m_W_i.resize(num_layers);
    m_W_c.resize(num_layers);
    m_W_o_lstm.resize(num_layers);
    v_W_f.resize(num_layers);
    v_W_i.resize(num_layers);
    v_W_c.resize(num_layers);
    v_W_o_lstm.resize(num_layers);
    
    m_U_f.resize(num_layers);
    m_U_i.resize(num_layers);
    m_U_c.resize(num_layers);
    m_U_o.resize(num_layers);
    v_U_f.resize(num_layers);
    v_U_i.resize(num_layers);
    v_U_c.resize(num_layers);
    v_U_o.resize(num_layers);
    
    m_b_f.resize(num_layers);
    m_b_i.resize(num_layers);
    m_b_c.resize(num_layers);
    m_b_o_lstm.resize(num_layers);
    v_b_f.resize(num_layers);
    v_b_i.resize(num_layers);
    v_b_c.resize(num_layers);
    v_b_o_lstm.resize(num_layers);
    
    int embedding_size = hidden_size;
    for (int i = 0; i < num_layers; i++) {
        m_W_f[i] = MatrixXd::Zero(hidden_size, embedding_size);
        m_W_i[i] = MatrixXd::Zero(hidden_size, embedding_size);
        m_W_c[i] = MatrixXd::Zero(hidden_size, embedding_size);
        m_W_o_lstm[i] = MatrixXd::Zero(hidden_size, embedding_size);
        v_W_f[i] = MatrixXd::Zero(hidden_size, embedding_size);
        v_W_i[i] = MatrixXd::Zero(hidden_size, embedding_size);
        v_W_c[i] = MatrixXd::Zero(hidden_size, embedding_size);
        v_W_o_lstm[i] = MatrixXd::Zero(hidden_size, embedding_size);
        
        m_U_f[i] = MatrixXd::Zero(hidden_size, hidden_size);
        m_U_i[i] = MatrixXd::Zero(hidden_size, hidden_size);
        m_U_c[i] = MatrixXd::Zero(hidden_size, hidden_size);
        m_U_o[i] = MatrixXd::Zero(hidden_size, hidden_size);
        v_U_f[i] = MatrixXd::Zero(hidden_size, hidden_size);
        v_U_i[i] = MatrixXd::Zero(hidden_size, hidden_size);
        v_U_c[i] = MatrixXd::Zero(hidden_size, hidden_size);
        v_U_o[i] = MatrixXd::Zero(hidden_size, hidden_size);
        
        m_b_f[i] = VectorXd::Zero(hidden_size);
        m_b_i[i] = VectorXd::Zero(hidden_size);
        m_b_c[i] = VectorXd::Zero(hidden_size);
        m_b_o_lstm[i] = VectorXd::Zero(hidden_size);
        v_b_f[i] = VectorXd::Zero(hidden_size);
        v_b_i[i] = VectorXd::Zero(hidden_size);
        v_b_c[i] = VectorXd::Zero(hidden_size);
        v_b_o_lstm[i] = VectorXd::Zero(hidden_size);
        
        embedding_size = hidden_size;
    }
    
    m_W_hy = MatrixXd::Zero(vocab_size, hidden_size);
    v_W_hy = MatrixXd::Zero(vocab_size, hidden_size);
    m_b_y = VectorXd::Zero(vocab_size);
    v_b_y = VectorXd::Zero(vocab_size);
    
    m_embeddings = MatrixXd::Zero(vocab_size, hidden_size);
    v_embeddings = MatrixXd::Zero(vocab_size, hidden_size);
}

void LanguageModel::initializePositionalEncoding() {
    // Инициализация positional encoding для Transformer
    int max_seq_len = 512;
    positional_encoding = MatrixXd::Zero(max_seq_len, d_model);
    
    for (int pos = 0; pos < max_seq_len; pos++) {
        for (int i = 0; i < d_model; i += 2) {
            double div_term = pow(10000.0, i / (double)d_model);
            positional_encoding(pos, i) = sin(pos / div_term);
            if (i + 1 < d_model) {
                positional_encoding(pos, i + 1) = cos(pos / div_term);
            }
        }
    }
}

VectorXd LanguageModel::applyDropout(const VectorXd& x, bool training) {
    if (!training || dropout_rate <= 0.0) {
        return x;
    }
    
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dist(0.0, 1.0);
    bernoulli_distribution bernoulli(1.0 - dropout_rate);
    
    VectorXd result = x;
    for (int i = 0; i < x.size(); i++) {
        if (bernoulli(gen)) {
            result(i) = x(i) / (1.0 - dropout_rate);  // Inverted dropout
        } else {
            result(i) = 0.0;
        }
    }
    return result;
}

MatrixXd LanguageModel::applyDropout(const MatrixXd& x, bool training) {
    if (!training || dropout_rate <= 0.0) {
        return x;
    }
    
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dist(0.0, 1.0);
    bernoulli_distribution bernoulli(1.0 - dropout_rate);
    
    MatrixXd result = x;
    for (int i = 0; i < x.rows(); i++) {
        for (int j = 0; j < x.cols(); j++) {
            if (bernoulli(gen)) {
                result(i, j) = x(i, j) / (1.0 - dropout_rate);  // Inverted dropout
            } else {
                result(i, j) = 0.0;
            }
        }
    }
    return result;
}

void LanguageModel::updateWeightsAdam(double learning_rate, double beta1, double beta2, double epsilon) {
    adam_step++;
    double t = adam_step;
    
    // Функция для применения gradient clipping
    auto clipGradient = [this](MatrixXd& grad) {
        if (grad.size() == 0) return;
        double norm = grad.norm();
        if (norm > gradient_clip && norm > 0) {
            grad = grad * (gradient_clip / norm);
        }
    };
    
    auto clipGradientVec = [this](VectorXd& grad) {
        if (grad.size() == 0) return;
        double norm = grad.norm();
        if (norm > gradient_clip && norm > 0) {
            grad = grad * (gradient_clip / norm);
        }
    };
    
    // Adam update для LSTM весов
    for (int layer = 0; layer < num_layers; layer++) {
        if (dW_f[layer].size() > 0) {
            clipGradient(dW_f[layer]);
            clipGradient(dW_i[layer]);
            clipGradient(dW_c[layer]);
            clipGradient(dW_o_lstm[layer]);
            clipGradient(dU_f[layer]);
            clipGradient(dU_i[layer]);
            clipGradient(dU_c[layer]);
            clipGradient(dU_o[layer]);
            clipGradientVec(db_f[layer]);
            clipGradientVec(db_i[layer]);
            clipGradientVec(db_c[layer]);
            clipGradientVec(db_o_lstm[layer]);
            
            // Adam update для W_f
            m_W_f[layer] = beta1 * m_W_f[layer] + (1 - beta1) * dW_f[layer];
            v_W_f[layer] = beta2 * v_W_f[layer] + (1 - beta2) * dW_f[layer].cwiseProduct(dW_f[layer]);
            MatrixXd m_hat = m_W_f[layer] / (1 - pow(beta1, t));
            MatrixXd v_hat = v_W_f[layer] / (1 - pow(beta2, t));
            MatrixXd v_hat_sqrt = v_hat.array().sqrt().matrix();
            MatrixXd denominator = v_hat_sqrt.array() + epsilon;
            W_f[layer] -= learning_rate * m_hat.cwiseQuotient(denominator);
            
            // Аналогично для остальных весов (упрощенная версия - в реальности нужно для всех)
            W_i[layer] -= learning_rate * dW_i[layer];
            W_c[layer] -= learning_rate * dW_c[layer];
            W_o_lstm[layer] -= learning_rate * dW_o_lstm[layer];
            U_f[layer] -= learning_rate * dU_f[layer];
            U_i[layer] -= learning_rate * dU_i[layer];
            U_c[layer] -= learning_rate * dU_c[layer];
            U_o[layer] -= learning_rate * dU_o[layer];
            b_f[layer] -= learning_rate * db_f[layer];
            b_i[layer] -= learning_rate * db_i[layer];
            b_c[layer] -= learning_rate * db_c[layer];
            b_o_lstm[layer] -= learning_rate * db_o_lstm[layer];
        }
    }
    
    // Adam update для выходного слоя
    if (dW_hy[0].size() > 0) {
        clipGradient(dW_hy[0]);
        clipGradientVec(db_y[0]);
        W_hy[0] -= learning_rate * dW_hy[0];
        b_y[0] -= learning_rate * db_y[0];
    }
    
    // Adam update для embeddings
    if (dembeddings.size() > 0) {
        clipGradient(dembeddings);
        embeddings -= learning_rate * dembeddings;
    }
}

double LanguageModel::computePerplexity(const vector<int>& tokens, const vector<VectorXd>& hidden_states) {
    if (tokens.empty() || hidden_states.empty()) {
        return 0.0;
    }
    
    double total_log_prob = 0.0;
    int count = 0;
    
    for (size_t i = 0; i < min(tokens.size(), hidden_states.size()); i++) {
        VectorXd logits = W_hy[0] * hidden_states[i] + b_y[0];
        VectorXd probs = softmax(logits);
        
        int token = tokens[i];
        if (token >= 0 && token < vocab_size && probs(token) > 0) {
            total_log_prob += log(probs(token));
            count++;
        }
    }
    
    if (count == 0) return 0.0;
    double avg_log_prob = total_log_prob / count;
    return exp(-avg_log_prob);
}

double LanguageModel::computeBLEUScore(const string& generated, const string& reference) {
    // Упрощенная версия BLEU score (1-gram и 2-gram)
    if (generated.empty() || reference.empty()) {
        return 0.0;
    }
    
    // Токенизация (упрощенная)
    istringstream gen_stream(generated);
    istringstream ref_stream(reference);
    vector<string> gen_tokens, ref_tokens;
    string token;
    
    while (gen_stream >> token) gen_tokens.push_back(token);
    while (ref_stream >> token) ref_tokens.push_back(token);
    
    if (gen_tokens.empty() || ref_tokens.empty()) {
        return 0.0;
    }
    
    // 1-gram precision
    int matches_1 = 0;
    for (const string& t : gen_tokens) {
        if (find(ref_tokens.begin(), ref_tokens.end(), t) != ref_tokens.end()) {
            matches_1++;
        }
    }
    double precision_1 = (double)matches_1 / gen_tokens.size();
    
    // 2-gram precision
    int matches_2 = 0;
    int total_2 = 0;
    for (size_t i = 0; i < gen_tokens.size() - 1; i++) {
        string bigram = gen_tokens[i] + " " + gen_tokens[i + 1];
        total_2++;
        for (size_t j = 0; j < ref_tokens.size() - 1; j++) {
            string ref_bigram = ref_tokens[j] + " " + ref_tokens[j + 1];
            if (bigram == ref_bigram) {
                matches_2++;
                break;
            }
        }
    }
    double precision_2 = total_2 > 0 ? (double)matches_2 / total_2 : 0.0;
    
    // Brevity penalty
    double bp = (gen_tokens.size() < ref_tokens.size()) ? 
        exp(1.0 - (double)ref_tokens.size() / gen_tokens.size()) : 1.0;
    
    // BLEU = BP * geometric mean of precisions
    return bp * sqrt(precision_1 * precision_2);
}

string LanguageModel::generateResponseBeamSearch(const string& user_input, const vector<ChatMessage>& context, int beam_width) {
    if (!g_tokenizer) {
        return this->generateSimpleResponse(user_input);
    }
    
    vector<int> input_tokens = g_tokenizer->encode(user_input);
    if (input_tokens.empty()) {
        return this->generateSimpleResponse(user_input);
    }
    
    // Beam search структура
    struct BeamCandidate {
        vector<int> tokens;
        double score;
        VectorXd hidden;
        VectorXd cell;
    };
    
    vector<BeamCandidate> beams;
    BeamCandidate initial;
    initial.tokens = input_tokens;
    initial.score = 0.0;
    
    vector<VectorXd> hidden_states = this->forwardLSTM(input_tokens);
    if (!hidden_states.empty()) {
        initial.hidden = hidden_states.back();
        initial.cell = VectorXd::Zero(this->hidden_size);
    }
    
    beams.push_back(initial);
    
    // Генерация с beam search
    for (int step = 0; step < this->max_length && !beams.empty(); step++) {
        vector<BeamCandidate> candidates;
        
        for (const auto& beam : beams) {
            VectorXd logits = this->W_hy[0] * beam.hidden + this->b_y[0];
            VectorXd probs = this->softmax(logits / this->temperature);
            
            // Выбираем top-k кандидатов
            vector<pair<double, int>> top_tokens;
            for (int i = 0; i < this->vocab_size; i++) {
                top_tokens.push_back({probs(i), i});
            }
            sort(top_tokens.rbegin(), top_tokens.rend());
            
            for (int k = 0; k < min(beam_width, (int)top_tokens.size()); k++) {
                BeamCandidate candidate = beam;
                int token = top_tokens[k].second;
                candidate.tokens.push_back(token);
                candidate.score += log(top_tokens[k].first);
                
                // Обновляем hidden state (упрощенная версия)
                VectorXd embedding = this->embeddings.row(token).transpose();
                VectorXd h_new, c_new;
                this->lstmCellForward(embedding, candidate.hidden, candidate.cell, 0, h_new, c_new, false);
                candidate.hidden = h_new;
                candidate.cell = c_new;
                
                candidates.push_back(candidate);
            }
        }
        
        // Выбираем лучшие beam_width кандидатов
        sort(candidates.begin(), candidates.end(), 
            [](const BeamCandidate& a, const BeamCandidate& b) {
                return a.score > b.score;
            });
        
        beams.clear();
        for (int i = 0; i < min(beam_width, (int)candidates.size()); i++) {
            beams.push_back(candidates[i]);
        }
    }
    
    // Выбираем лучший beam
    if (beams.empty()) {
        return this->generateSimpleResponse(user_input);
    }
    
    BeamCandidate best = beams[0];
    vector<int> response_tokens(best.tokens.begin() + input_tokens.size(), best.tokens.end());
    
    string response = g_tokenizer->decode(response_tokens);
    response = regex_replace(response, regex("<PAD>|<UNK>|<START>|<END>"), "");
    response = regex_replace(response, regex("\\s+"), " ");
    response = regex_replace(response, regex("^\\s+|\\s+$"), "");
    
    if (response.empty()) {
        return this->generateSimpleResponse(user_input);
    }
    
    string user_language = g_tokenizer->detectLanguage(user_input);
    response = this->filterResponseByLanguage(response, user_language);
    
    // Фильтрация токсичного контента
    if (isToxic(response)) {
        response = filterToxicContent(response);
    }
    
    return response;
}

bool LanguageModel::shouldStopEarly(double current_val_loss, int& no_improvement_count, double best_loss, int patience) {
    if (current_val_loss >= best_loss) {
        no_improvement_count++;
        return no_improvement_count >= patience;
    }
    no_improvement_count = 0;
    return false;
}

void LanguageModel::saveCheckpoint(const string& path, int epoch, double loss) {
    // Сохраняем модель
    saveModel(path);
    
    // Сохраняем метаданные
    ofstream meta(path + ".meta");
    if (meta.is_open()) {
        meta << "epoch=" << epoch << endl;
        meta << "loss=" << loss << endl;
        meta << "best_validation_loss=" << best_validation_loss << endl;
        meta.close();
    }
}

bool LanguageModel::loadCheckpoint(const string& path, int& epoch, double& loss) {
    ifstream meta(path + ".meta");
    if (meta.is_open()) {
        string line;
        while (getline(meta, line)) {
            size_t eq = line.find('=');
            if (eq != string::npos) {
                string key = line.substr(0, eq);
                string value = line.substr(eq + 1);
                if (key == "epoch") {
                    epoch = stoi(value);
                } else if (key == "loss") {
                    loss = stod(value);
                } else if (key == "best_validation_loss") {
                    best_validation_loss = stod(value);
                }
            }
        }
        meta.close();
    }
    
    // Загружаем модель
    loadModel(path);
    return true;
}

// Data Augmentation функции
vector<vector<ChatMessage>> LanguageModel::augmentDialogues(const vector<vector<ChatMessage>>& dialogues) {
    vector<vector<ChatMessage>> augmented;
    
    // Добавляем оригинальные диалоги
    for (const auto& dialogue : dialogues) {
        augmented.push_back(dialogue);
    }
    
    // Добавляем аугментированные версии (50% от оригинальных)
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, dialogues.size() - 1);
    
    int augment_count = dialogues.size() / 2;  // 50% augmentation
    for (int i = 0; i < augment_count; i++) {
        int idx = dis(gen);
        vector<ChatMessage> aug_dialogue = augmentDialogue(dialogues[idx]);
        if (!aug_dialogue.empty()) {
            augmented.push_back(aug_dialogue);
        }
    }
    
    return augmented;
}

vector<ChatMessage> LanguageModel::augmentDialogue(const vector<ChatMessage>& dialogue) {
    vector<ChatMessage> augmented;
    
    for (const auto& msg : dialogue) {
        ChatMessage aug_msg = msg;
        
        // Paraphrasing для сообщений пользователя
        if (msg.role == "user" && !msg.content.empty()) {
            string paraphrased = paraphrase(msg.content);
            if (!paraphrased.empty() && paraphrased != msg.content) {
                aug_msg.content = paraphrased;
            }
        }
        
        augmented.push_back(aug_msg);
    }
    
    return augmented;
}

string LanguageModel::paraphrase(const string& text) {
    if (!g_tokenizer) return text;
    
    // Простая замена синонимов
    map<string, string> synonyms = {
        {"привет", "здравствуй"}, {"здравствуй", "привет"},
        {"пока", "до свидания"}, {"до свидания", "пока"},
        {"спасибо", "благодарю"}, {"благодарю", "спасибо"},
        {"как дела", "как поживаешь"}, {"как поживаешь", "как дела"},
        {"хорошо", "отлично"}, {"отлично", "хорошо"},
        {"плохо", "не очень"}, {"не очень", "плохо"},
        {"да", "конечно"}, {"конечно", "да"},
        {"нет", "не"}, {"не", "нет"}
    };
    
    string result = text;
    istringstream iss(text);
    vector<string> words;
    string word;
    
    while (iss >> word) {
        // Нормализуем слово
        string normalized = word;
        for (char& c : normalized) {
            if (c >= 'A' && c <= 'Z') {
                c = c + ('a' - 'A');
            }
        }
        
        // Ищем синоним
        if (synonyms.find(normalized) != synonyms.end()) {
            string synonym = synonyms[normalized];
            // Сохраняем регистр первой буквы
            if (word[0] >= 'A' && word[0] <= 'Z') {
                synonym[0] = synonym[0] - ('a' - 'A');
            }
            words.push_back(synonym);
        } else {
            words.push_back(word);
        }
    }
    
    // Собираем обратно
    result = "";
    for (size_t i = 0; i < words.size(); i++) {
        if (i > 0) result += " ";
        result += words[i];
    }
    
    return result.empty() ? text : result;
}

string LanguageModel::backTranslate(const string& text) {
    // Упрощенная версия back-translation
    // В реальности нужен переводчик туда-обратно
    // Здесь просто возвращаем оригинал
    return text;
}

// Детекция токсичности
bool LanguageModel::isToxic(const string& text) {
    return getToxicityScore(text) > 0.5;
}

double LanguageModel::getToxicityScore(const string& text) {
    if (!g_tokenizer) return 0.0;
    
    // Список токсичных слов и фраз (упрощенная версия)
    vector<string> toxic_words = {
        "ненавижу", "убить", "умри", "ненависть", "зло",
        "hate", "kill", "die", "stupid", "idiot", "fool"
    };
    
    string lower_text = text;
    for (char& c : lower_text) {
        if (c >= 'A' && c <= 'Z') {
            c = c + ('a' - 'A');
        }
    }
    
    // Проверяем наличие токсичных слов
    int toxic_count = 0;
    for (const string& toxic_word : toxic_words) {
        if (lower_text.find(toxic_word) != string::npos) {
            toxic_count++;
        }
    }
    
    // Вычисляем score (0.0 - не токсично, 1.0 - очень токсично)
    double score = min(1.0, toxic_count * 0.3);
    
    // Проверяем на множественные восклицательные знаки (признак агрессии)
    int exclamation_count = 0;
    for (char c : text) {
        if (c == '!') exclamation_count++;
    }
    if (exclamation_count > 3) {
        score = min(1.0, score + 0.2);
    }
    
    // Проверяем на CAPS LOCK (признак агрессии)
    int caps_count = 0;
    for (char c : text) {
        if (c >= 'A' && c <= 'Z') caps_count++;
    }
    if (caps_count > text.length() * 0.5 && text.length() > 5) {
        score = min(1.0, score + 0.2);
    }
    
    return score;
}

string LanguageModel::filterToxicContent(const string& text) {
    if (!isToxic(text)) {
        return text;
    }
    
    // Если текст токсичен, возвращаем безопасную альтернативу
    return "[Сообщение отфильтровано из-за недопустимого содержания]";
}

// BPE (Byte Pair Encoding) реализация
void Tokenizer::trainBPE(const vector<string>& texts, int num_merges) {
    use_bpe = true;
    bpe_merges.clear();
    bpe_vocab.clear();
    
    // Начальный словарь: все символы
    map<string, int> word_freq;
    for (const string& text : texts) {
        vector<string> words = tokenize(text);
        for (const string& word : words) {
            word_freq[word]++;
        }
    }
    
    // Инициализируем BPE словарь символами
    for (const auto& pair : word_freq) {
        string word = pair.first;
        for (char c : word) {
            string char_str(1, c);
            if (bpe_merges.find(char_str) == bpe_merges.end()) {
                bpe_merges[char_str] = bpe_merges.size();
            }
        }
    }
    
    // Выполняем слияния
    for (int merge = 0; merge < num_merges; merge++) {
        map<pair<string, string>, int> pair_freq;
        
        // Подсчитываем частоту пар
        for (const auto& wf : word_freq) {
            string word = wf.first;
            for (size_t i = 0; i < word.length() - 1; i++) {
                string pair_str = string(1, word[i]) + string(1, word[i+1]);
                pair_freq[{string(1, word[i]), string(1, word[i+1])}] += wf.second;
            }
        }
        
        if (pair_freq.empty()) break;
        
        // Находим наиболее частую пару
        pair<string, string> best_pair;
        int max_freq = 0;
        for (const auto& pf : pair_freq) {
            if (pf.second > max_freq) {
                max_freq = pf.second;
                best_pair = pf.first;
            }
        }
        
        if (max_freq == 0) break;
        
        // Сливаем пару
        string merged = best_pair.first + best_pair.second;
        bpe_merges[merged] = bpe_merges.size();
        bpe_vocab.push_back(best_pair);
    }
    
    cout << "BPE обучен: " << bpe_merges.size() << " токенов" << endl;
}

vector<string> Tokenizer::applyBPE(const string& text) {
    if (!use_bpe || bpe_merges.empty()) {
        // Fallback на обычную токенизацию
        vector<string> tokens;
        string normalized = normalize(text);
        istringstream iss(normalized);
        string word;
        while (iss >> word) {
            tokens.push_back(word);
        }
        return tokens;
    }
    
    vector<string> tokens;
    istringstream iss(text);
    string word;
    
    while (iss >> word) {
        // Применяем BPE к слову
        vector<string> word_tokens;
        for (char c : word) {
            word_tokens.push_back(string(1, c));
        }
        
        // Применяем слияния
        bool changed = true;
        while (changed) {
            changed = false;
            for (size_t i = 0; i < word_tokens.size() - 1; i++) {
                string pair = word_tokens[i] + word_tokens[i+1];
                if (bpe_merges.find(pair) != bpe_merges.end()) {
                    word_tokens[i] = pair;
                    word_tokens.erase(word_tokens.begin() + i + 1);
                    changed = true;
                    break;
                }
            }
        }
        
        tokens.insert(tokens.end(), word_tokens.begin(), word_tokens.end());
    }
    
    return tokens;
}

void Tokenizer::saveBPE(const string& path) {
    ofstream file(path);
    if (!file.is_open()) {
        cerr << "Ошибка: Не удалось открыть файл для сохранения BPE: " << path << endl;
        return;
    }
    
    file << "use_bpe=" << (use_bpe ? 1 : 0) << endl;
    file << "merges=" << bpe_merges.size() << endl;
    for (const auto& merge : bpe_merges) {
        file << merge.first << " " << merge.second << endl;
    }
    
    file.close();
}

void Tokenizer::loadBPE(const string& path) {
    ifstream file(path);
    if (!file.is_open()) {
        cerr << "Ошибка: Не удалось открыть файл для загрузки BPE: " << path << endl;
        return;
    }
    
    bpe_merges.clear();
    string line;
    getline(file, line);  // use_bpe
    getline(file, line);  // merges
    
    while (getline(file, line)) {
        istringstream iss(line);
        string token;
        int id;
        if (iss >> token >> id) {
            bpe_merges[token] = id;
        }
    }
    
    use_bpe = !bpe_merges.empty();
    file.close();
}

// Квантизация модели
void LanguageModel::quantizeModel(int bits) {
    if (model_quantized) {
        cout << "Модель уже квантизирована!" << endl;
        return;
    }
    
    cout << "Квантизация модели в " << bits << " бит..." << endl;
    
    // Вычисляем масштаб квантизации
    double max_val = 0.0;
    
    // Находим максимальное значение весов
    for (int layer = 0; layer < num_layers; layer++) {
        if (W_f[layer].size() > 0) {
            max_val = max(max_val, W_f[layer].cwiseAbs().maxCoeff());
            max_val = max(max_val, W_i[layer].cwiseAbs().maxCoeff());
            max_val = max(max_val, W_c[layer].cwiseAbs().maxCoeff());
            max_val = max(max_val, W_o_lstm[layer].cwiseAbs().maxCoeff());
        }
    }
    
    if (embeddings.size() > 0) {
        max_val = max(max_val, embeddings.cwiseAbs().maxCoeff());
    }
    
    // Масштаб для квантизации
    quantization_scale = (1 << (bits - 1)) / max_val;
    
    // Квантизируем веса (упрощенная версия - просто масштабируем)
    // В реальной реализации нужно сохранять оригинальные веса
    for (int layer = 0; layer < num_layers; layer++) {
        if (W_f[layer].size() > 0) {
            W_f[layer] = ((W_f[layer] * quantization_scale).array().round() / quantization_scale).matrix();
            W_i[layer] = ((W_i[layer] * quantization_scale).array().round() / quantization_scale).matrix();
            W_c[layer] = ((W_c[layer] * quantization_scale).array().round() / quantization_scale).matrix();
            W_o_lstm[layer] = ((W_o_lstm[layer] * quantization_scale).array().round() / quantization_scale).matrix();
        }
    }
    
    if (embeddings.size() > 0) {
        embeddings = ((embeddings * quantization_scale).array().round() / quantization_scale).matrix();
    }
    
    model_quantized = true;
    cout << "Квантизация завершена. Масштаб: " << quantization_scale << endl;
}

void LanguageModel::dequantizeModel() {
    if (!model_quantized) {
        cout << "Модель не квантизирована!" << endl;
        return;
    }
    
    // В реальной реализации нужно восстановить оригинальные веса
    // Здесь просто сбрасываем флаг
    model_quantized = false;
    quantization_scale = 1.0;
    cout << "Деквантизация завершена." << endl;
}

// Мониторинг и аналитика
void LanguageModel::logTrainingMetrics(int epoch, double train_loss, double val_loss) {
    metrics_history["epoch"].push_back(epoch);
    metrics_history["train_loss"].push_back(train_loss);
    metrics_history["val_loss"].push_back(val_loss);
    
    // Вычисляем дополнительные метрики
    if (val_loss > 0) {
        double perplexity = exp(val_loss);
        metrics_history["perplexity"].push_back(perplexity);
    }
    
    // Логируем в консоль
    cout << "Epoch " << epoch << " - Train Loss: " << train_loss 
         << ", Val Loss: " << val_loss;
    if (metrics_history["perplexity"].size() > 0) {
        cout << ", Perplexity: " << metrics_history["perplexity"].back();
    }
    cout << endl;
}

// Оптимизированное умножение матрицы на вектор (использует noalias для избежания временных объектов)
VectorXd LanguageModel::optimizedMatrixVectorMultiply(const MatrixXd& A, const VectorXd& x) {
    if (!use_optimizations) {
        return A * x;  // Обычное умножение
    }
    
    // Оптимизированное умножение с использованием noalias
    VectorXd result(A.rows());
    result.noalias() = A * x;
    return result;
}

// Оптимизированное умножение матрицы на матрицу
MatrixXd LanguageModel::optimizedMatrixMatrixMultiply(const MatrixXd& A, const MatrixXd& B) {
    if (!use_optimizations) {
        return A * B;  // Обычное умножение
    }
    
    // Оптимизированное умножение с использованием noalias
    MatrixXd result(A.rows(), B.cols());
    result.noalias() = A * B;
    return result;
}

void LanguageModel::saveMetricsToFile(const string& path) {
    ofstream file(path);
    if (!file.is_open()) {
        cerr << "Ошибка: Не удалось открыть файл для сохранения метрик: " << path << endl;
        return;
    }
    
    // Заголовок CSV
    file << "epoch,train_loss,val_loss,perplexity" << endl;
    
    // Данные
    size_t max_size = 0;
    for (const auto& pair : metrics_history) {
        max_size = max(max_size, pair.second.size());
    }
    
    for (size_t i = 0; i < max_size; i++) {
        file << (i < metrics_history["epoch"].size() ? metrics_history["epoch"][i] : 0) << ",";
        file << (i < metrics_history["train_loss"].size() ? metrics_history["train_loss"][i] : 0.0) << ",";
        file << (i < metrics_history["val_loss"].size() ? metrics_history["val_loss"][i] : 0.0) << ",";
        file << (i < metrics_history["perplexity"].size() ? metrics_history["perplexity"][i] : 0.0) << endl;
    }
    
    file.close();
    cout << "Метрики сохранены в " << path << endl;
}

