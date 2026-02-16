#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdio.h>
#include <commdlg.h>
#include <shlobj.h>   // SHBrowseForFolder, SHGetPathFromIDList

static WNDPROC g_origWndProc = nullptr;
static bool g_closeDisabled = false;
static LRESULT CALLBACK NoCloseWndProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    if (msg == WM_CLOSE) return 0;  // Игнорируем закрытие
    return CallWindowProc(g_origWndProc, hwnd, msg, wp, lp);
}
static HWND findOpenCVWindow(const char* winName) {
    HWND h = FindWindowA(NULL, winName);
    if (h) return h;
    HWND found = NULL;
    EnumWindows([](HWND hwnd, LPARAM lp) -> BOOL {
        char buf[256];
        if (GetWindowTextA(hwnd, buf, sizeof(buf)) && strstr(buf, "Image Classifier")) {
            *(HWND*)lp = hwnd;
            return FALSE;
        }
        return TRUE;
    }, (LPARAM)&found);
    return found;
}
static void disableWindowClose(const char* winName) {
    if (g_closeDisabled) return;
    for (int retry = 0; retry < 30; retry++) {
        HWND hwnd = findOpenCVWindow(winName);
        if (hwnd) {
            HWND root = GetAncestor(hwnd, GA_ROOT);
            if (!root) root = hwnd;
            HMENU sysMenu = GetSystemMenu(root, FALSE);
            if (sysMenu) RemoveMenu(sysMenu, SC_CLOSE, MF_BYCOMMAND);
            g_origWndProc = (WNDPROC)SetWindowLongPtr(root, GWLP_WNDPROC, (LONG_PTR)NoCloseWndProc);
            g_closeDisabled = true;
            return;
        }
        Sleep(50);
    }
}
static LONG WINAPI crashHandler(EXCEPTION_POINTERS* p) {
    HANDLE h = CreateFileA("classifier_crash.log", FILE_APPEND_DATA, FILE_SHARE_READ, NULL, OPEN_ALWAYS, 0, NULL);
    if (h != INVALID_HANDLE_VALUE) {
        char buf[128];
        int len = sprintf_s(buf, "=== CRASH code=%lu\n", (unsigned long)p->ExceptionRecord->ExceptionCode);
        DWORD written;
        WriteFile(h, buf, len, &written, NULL);
        CloseHandle(h);
    }
    return EXCEPTION_EXECUTE_HANDLER;
}
#endif

#include "UniversalImageClassifier.h"
#include "ImageDownloader.h"
#include "CudaAccelerator.h"
#include "TextRenderer.h"
#include "Config.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <thread>
#include <chrono>
#include <future>
#include <vector>
#include <string>
#include <mutex>
#include <iomanip>
#include <sstream>
#include <map>
#include <set>
#include <algorithm>
#include <ctime>
#include <exception>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

// --- UI Constants --- Military Camouflage Theme
const Scalar BG_COLOR(20, 30, 20);  // Dark olive green camouflage
const Scalar BUTTON_COLOR(40, 60, 40);  // Dark green
const Scalar BUTTON_HOVER_COLOR(60, 80, 60);  // Lighter green
const Scalar TEXT_COLOR(173, 255, 47);  // Yellow-green military color
const Scalar ACCENT_COLOR(255, 140, 0);  // Dark orange military
const Scalar PROGRESS_BG(30, 45, 30);  // Dark green
const Scalar PROGRESS_FILL(0, 200, 0);  // Bright green
const Scalar TABLE_HEADER_COLOR(30, 50, 30);  // Dark green
const Scalar TABLE_ROW_COLOR1(25, 40, 25);  // Very dark green
const Scalar TABLE_ROW_COLOR2(35, 55, 35);  // Medium dark green
const Scalar WARNING_COLOR(0, 255, 255);  // Yellow (BGR)

// --- Global State ---
enum AppState { STATE_MAIN, STATE_TRAINING };
AppState g_app_state = STATE_MAIN;

UniversalImageClassifier* g_classifier = nullptr;
Mat g_main_ui;
Mat g_current_image;
string g_status_text = "Ready";
string g_classification_result = "";
double g_classification_confidence = 0.0;
int g_predicted_class_id = -1;
vector<NeuralNetwork::ClassMetrics> g_class_metrics;
double g_macro_f1 = 0.0;

bool g_training_active = false;
double g_train_progress = 0.0;
double g_problem_class_progress = 0.0;
double g_quality_control_progress = 0.0;
int g_stage4_epoch_count = 0;
int g_current_epoch = 0;
int g_current_stage = 0;
int g_total_epochs = 0;
static const int g_stage3_epochs = 15;  // Stage 3 epochs (stage 2 = g_total_epochs)
static const int g_stage4_total_epochs = 150;  // 75 epochs x 2 cycles
string g_training_status = "Ready to train";
double g_current_loss = 0.0;
double g_current_accuracy = 0.0;
int g_processed_images = 0;
int g_total_images = 0;
chrono::steady_clock::time_point g_training_start_time;
int g_batch_size = 8;
double g_learning_rate = 0.001;

// Training statistics
vector<TrainingStats> g_training_history;
vector<NeuralNetwork::ClassMetrics> g_final_class_metrics;
vector<int> g_problem_classes;
double g_final_accuracy = 0.0;
double g_final_macro_f1 = 0.0;
double g_final_weighted_f1 = 0.0;
bool g_training_completed = false;

// Fast / Full training mode
bool g_fast_mode = true;

// CLI: автостарт обучения (--train "path")
bool g_auto_start_training = false;

// Saved metrics for display on main screen (from last training or last_metrics.json)
double g_display_accuracy = 0.0;
double g_display_f1 = 0.0;
int g_display_num_classes = 0;

// Model base path (where model was loaded from) — used for all saves
static string g_model_base;

mutex g_ui_mutex;
Point g_mouse_pos(-1, -1);

// Лог обучения (консоль + файл), UI не заменяется
static ofstream g_training_log;
static mutex g_log_mutex;
static int g_last_logged_epoch = -1;

// Window dimensions
const int WINDOW_WIDTH = 1400;
const int WINDOW_HEIGHT = 1050;

// Button definitions (main screen - updated in drawMainScreen)
Rect g_button_load_photo(50, 100, 200, 50);
Rect g_button_training(300, 100, 200, 50);
int g_metrics_scroll_offset = 0;
Rect g_button_back(50, 920, 150, 40);
Rect g_button_start_training(250, 920, 200, 40);
Rect g_button_select_dir(500, 920, 270, 40);
Rect g_button_mode(800, 920, 150, 40);

// Error notifications (dismissible toasts at bottom)
struct Notification {
    int id;
    string message;
    Rect close_rect;
    bool closed;
};
static vector<Notification> g_notifications;
static int g_next_notification_id = 1;
static mutex g_notifications_mutex;
static const int NOTIFICATION_HEIGHT = 40;
static const int MAX_NOTIFICATIONS = 10;
static const int MAX_VISIBLE_NOTIFICATIONS = 5;

// Training dataset directory (can be changed by user)
string g_training_dir = "training_images";
static vector<string> g_cached_paths;
static vector<int> g_cached_labels;
static string g_cached_dir;
static bool g_cache_valid = false;

// --- Prototypes ---
void drawMainScreen();
void drawTrainingScreen();
void drawButton(Mat& img, Rect rect, string text, bool hover = false);
void drawCircleProgress(Mat& img, Point center, int radius, double progress, bool complete, const string& label);
void drawClassStatisticsTable(Mat& img, int start_x, int start_y, int table_width = -1);
void drawImageWithResult(Mat& img, int start_x, int start_y, int width, int height);
void drawResultAndParams(Mat& img, int start_x, int start_y);
void drawBPLA(Mat& img, Point position, Size size);
void onMouse(int event, int x, int y, int flags, void* userdata);
string openFileDialog();
string openFolderDialog();
void startTraining();
void loadImageAndClassify(const string& path);
void reclassifyCurrentImage();  // Re-classify g_current_image (e.g. after training to update class names)
string formatTime(chrono::steady_clock::duration duration);
void saveBestModelMetrics();
void loadBestModelMetrics();
void saveTrainingDir();
void loadTrainingDir(const char* exe_path = nullptr);
void ensureClassNamesFromTrainingDir();
void saveTrainingHistory();
void loadTrainingHistory();
void pushNotification(const string& message);
void drawNotifications(Mat& img);
void removeClosedNotifications();

// --- UI Helpers ---
void drawButton(Mat& img, Rect rect, string text, bool hover) {
    Scalar btn_color = hover ? BUTTON_HOVER_COLOR : BUTTON_COLOR;
    rectangle(img, rect, btn_color, -1);
    rectangle(img, rect, Scalar(100, 100, 100), 2);
    
    int baseline = 0;
    Size textSize = getTextSize(text, FONT_HERSHEY_SIMPLEX, 0.7, 2, &baseline);
    Point textPos(rect.x + (rect.width - textSize.width) / 2, 
                  rect.y + (rect.height + textSize.height) / 2);
    putText(img, text, textPos, FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2, LINE_AA);
}

void drawProgressBar(Mat& img, Rect rect, double progress) {
    rectangle(img, rect, PROGRESS_BG, -1);
    int fill_width = static_cast<int>(rect.width * max(0.0, min(1.0, progress)));
    if (fill_width > 0) {
        rectangle(img, Rect(rect.x, rect.y, fill_width, rect.height), PROGRESS_FILL, -1);
    }
    rectangle(img, rect, Scalar(100, 100, 100), 2);
    
    string pct = to_string(int(progress * 100)) + "%";
    int baseline = 0;
    Size textSize = getTextSize(pct, FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseline);
    Point textPos(rect.x + (rect.width - textSize.width) / 2, 
                  rect.y + (rect.height + textSize.height) / 2);
    putText(img, pct, textPos, FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2, LINE_AA);
}

void drawCircleProgress(Mat& img, Point center, int radius, double progress, bool complete, const string& label) {
    progress = max(0.0, min(1.0, progress));
    Scalar bg_color(25, 40, 25);
    Scalar fill_color = complete ? Scalar(0, 200, 0) : PROGRESS_FILL;
    Scalar border_color(80, 100, 80);
    circle(img, center, radius + 4, border_color, 2, LINE_AA);
    circle(img, center, radius, bg_color, -1, LINE_AA);
    if (progress > 0.001 || complete) {
        int thickness = radius / 4;
        if (thickness < 4) thickness = 4;
        double start_angle = 270;
        double sweep = 360.0 * (complete ? 1.0 : progress);
        ellipse(img, center, Size(radius - thickness/2, radius - thickness/2), 0, start_angle, start_angle + sweep, fill_color, thickness, LINE_AA);
    }
    string pct_str = complete ? "100%" : (to_string(int(progress * 100)) + "%");
    int baseline = 0;
    Size ts = getTextSize(pct_str, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
    putText(img, pct_str, Point(center.x - ts.width/2, center.y + ts.height/2), FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1, LINE_AA);
    if (!label.empty()) {
        int lb = 0;
        Size ls = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &lb);
        putText(img, label, Point(center.x - ls.width/2, center.y + radius + 25), FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1, LINE_AA);
    }
}

void logTraining(const string& msg) {
    cout << "[Train] " << msg << endl;
    cout.flush();
    lock_guard<mutex> lock(g_log_mutex);
    if (g_training_log.is_open()) {
        g_training_log << "[Train] " << msg << endl;
        g_training_log.flush();
    }
}

void pushNotification(const string& message) {
    if (message.empty()) return;
    lock_guard<mutex> lock(g_notifications_mutex);
    if (g_notifications.size() >= static_cast<size_t>(MAX_NOTIFICATIONS)) {
        g_notifications.erase(g_notifications.begin());
    }
    Notification n;
    n.id = g_next_notification_id++;
    n.message = message;
    n.close_rect = Rect(0, 0, 0, 0);
    n.closed = false;
    g_notifications.push_back(n);
}

void removeClosedNotifications() {
    lock_guard<mutex> lock(g_notifications_mutex);
    g_notifications.erase(
        remove_if(g_notifications.begin(), g_notifications.end(),
                 [](const Notification& n) { return n.closed; }),
        g_notifications.end());
}

void drawNotifications(Mat& img) {
    lock_guard<mutex> lock(g_notifications_mutex);
    const int margin = 50;
    const int notif_width = WINDOW_WIDTH - 2 * margin;
    const int bottom_start = WINDOW_HEIGHT - 50;
    int draw_count = 0;
    const int max_draw = min(static_cast<int>(g_notifications.size()), MAX_VISIBLE_NOTIFICATIONS);
    for (int i = static_cast<int>(g_notifications.size()) - 1; i >= 0 && draw_count < max_draw; --i) {
        Notification& n = g_notifications[i];
        if (n.closed) continue;
        int y = bottom_start - (draw_count + 1) * NOTIFICATION_HEIGHT - draw_count * 4;
        Rect notif_rect(margin, y, notif_width, NOTIFICATION_HEIGHT);
        rectangle(img, notif_rect, Scalar(20, 30, 50), -1);
        rectangle(img, notif_rect, Scalar(0, 0, 200), 2);
        int bl = 0;
        Size ts = getTextSize(n.message, FONT_HERSHEY_SIMPLEX, 0.55, 1, &bl);
        int text_max = notif_width - 50;
        string display_msg = n.message;
        if (ts.width > text_max) {
            while (getTextSize(display_msg + "...", FONT_HERSHEY_SIMPLEX, 0.55, 1, &bl).width > text_max && display_msg.size() > 3) {
                display_msg.pop_back();
            }
            display_msg += "...";
        }
        putText(img, display_msg, Point(margin + 8, y + NOTIFICATION_HEIGHT / 2 + 8), FONT_HERSHEY_SIMPLEX, 0.55, Scalar(200, 200, 255), 1, LINE_AA);
        const int close_size = 24;
        Rect close_rect(notif_rect.x + notif_rect.width - close_size - 4, y + (NOTIFICATION_HEIGHT - close_size) / 2, close_size, close_size);
        n.close_rect = close_rect;
        rectangle(img, close_rect, Scalar(60, 60, 80), -1);
        rectangle(img, close_rect, Scalar(150, 150, 150), 1);
        putText(img, "X", Point(close_rect.x + 6, close_rect.y + 18), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(200, 200, 200), 1, LINE_AA);
        draw_count++;
    }
}

string formatTime(chrono::steady_clock::duration duration) {
    auto hours = chrono::duration_cast<chrono::hours>(duration);
    duration -= hours;
    auto minutes = chrono::duration_cast<chrono::minutes>(duration);
    duration -= minutes;
    auto seconds = chrono::duration_cast<chrono::seconds>(duration);
    
    stringstream ss;
    if (hours.count() > 0) {
        ss << hours.count() << "h ";
    }
    if (minutes.count() > 0 || hours.count() > 0) {
        ss << minutes.count() << "m ";
    }
    ss << seconds.count() << "s";
    return ss.str();
}

static string getModelDir() {
    if (!g_model_base.empty()) {
        fs::path p(g_model_base);
        return p.parent_path().string();
    }
    return "";
}

void saveTrainingDir() {
    if (g_training_dir.empty()) return;
    string abs_path;
    try {
        abs_path = fs::absolute(g_training_dir).string();
    } catch (...) {
        abs_path = g_training_dir;
    }
    vector<string> paths;
    string model_dir = getModelDir();
    if (!model_dir.empty()) {
        paths.push_back((fs::path(model_dir) / "training_dir.txt").string());
    }
    paths.push_back("training_dir.txt");
    paths.push_back("x64/Release/training_dir.txt");
    for (const auto& p : paths) {
        try {
            ofstream f(p);
            if (f.is_open()) {
                f << abs_path;
                f.close();
                break;
            }
        } catch (...) {}
    }
}

void ensureClassNamesFromTrainingDir() {
    if (!g_classifier) return;
    auto list = g_classifier->getClassList();
    if (list.empty() || g_classifier->getClassName(0).find("Class ") != 0) return;
    vector<string> dirs_to_try;
    if (!g_training_dir.empty() && fs::exists(g_training_dir)) dirs_to_try.push_back(g_training_dir);
    string model_dir = getModelDir();
    if (!model_dir.empty()) dirs_to_try.push_back((fs::path(model_dir) / "training_images").string());
    dirs_to_try.push_back("training_images");
    dirs_to_try.push_back(".");
    for (const string& dir : dirs_to_try) {
        try {
            if (!fs::exists(dir)) continue;
            set<string> folder_names;
            for (const auto& entry : fs::directory_iterator(dir)) {
                if (entry.is_directory()) {
                    string fn = entry.path().filename().string();
                    if (fn != "training_images" && fn != "from_folder") folder_names.insert(fn);
                }
            }
            if (!folder_names.empty()) {
                vector<string> sorted_folders(folder_names.begin(), folder_names.end());
                sort(sorted_folders.begin(), sorted_folders.end());
                vector<string> names(sorted_folders.size());
                for (size_t i = 0; i < sorted_folders.size(); ++i) names[i] = sorted_folders[i];
                g_classifier->setClassNames(names);
                return;
            }
        } catch (...) { continue; }
    }
}

void loadTrainingDir(const char* exe_path) {
    vector<string> paths;
    try {
        paths.push_back((fs::current_path() / "training_dir.txt").string());
    } catch (...) {}
    paths.push_back("training_dir.txt");
    paths.push_back("./training_dir.txt");
    string model_dir = getModelDir();
    if (!model_dir.empty()) {
        paths.push_back((fs::path(model_dir) / "training_dir.txt").string());
    }
#ifdef _WIN32
    char exe_buf[MAX_PATH] = {0};
    if (GetModuleFileNameA(nullptr, exe_buf, MAX_PATH) && exe_buf[0]) {
        try {
            fs::path exe(exe_buf);
            if (exe.has_parent_path()) {
                paths.push_back((exe.parent_path() / "training_dir.txt").string());
            }
        } catch (...) {}
    }
    if (exe_path && exe_path[0]) {
        try {
            fs::path exe(exe_path);
            if (exe.is_absolute() && exe.has_parent_path()) {
                paths.push_back((exe.parent_path() / "training_dir.txt").string());
            }
        } catch (...) {}
    }
#endif
    paths.push_back("x64/Release/training_dir.txt");
    for (const auto& p : paths) {
        try {
            ifstream f(p);
            if (f.is_open()) {
                string line;
                if (getline(f, line) && !line.empty() && fs::exists(line)) {
                    g_training_dir = line;
                }
                f.close();
                break;
            }
        } catch (...) {}
    }
    if (g_training_dir.empty() || !fs::exists(g_training_dir)) {
        auto& cfg = Config::getInstance();
        vector<string> cfg_paths = {"config.json", "x64/Release/config.json"};
        if (!model_dir.empty()) cfg_paths.push_back((fs::path(model_dir) / "config.json").string());
#ifdef _WIN32
        if (exe_path && exe_path[0]) {
            try {
                fs::path exe(exe_path);
                if (exe.is_absolute() && exe.has_parent_path())
                    cfg_paths.push_back((exe.parent_path() / "config.json").string());
            } catch (...) {}
        }
#endif
        for (const auto& cp : cfg_paths) {
            if (cfg.load(cp)) {
                string cfg_dir = cfg.get("training_dir");
                if (!cfg_dir.empty() && fs::exists(cfg_dir)) {
                    g_training_dir = cfg_dir;
                    break;
                }
            }
        }
    }
}

void saveBestModelMetrics() {
    vector<string> paths;
    string model_dir = getModelDir();
    if (!model_dir.empty()) {
        paths.push_back((fs::path(model_dir) / "best_model_metrics.json").string());
    }
    paths.push_back("best_model_metrics.json");
    paths.push_back("x64/Release/best_model_metrics.json");
    for (const auto& p : paths) {
        try {
            ofstream f(p);
            if (f.is_open()) {
                f << "{\"accuracy\":" << fixed << setprecision(4) << g_final_accuracy
                  << ",\"macro_f1\":" << g_final_macro_f1
                  << ",\"num_classes\":" << g_display_num_classes
                  << ",\"metrics\":[";
                for (size_t i = 0; i < g_final_class_metrics.size(); ++i) {
                    const auto& m = g_final_class_metrics[i];
                    if (i > 0) f << ",";
                    f << "{\"class_id\":" << m.class_id << ",\"precision\":" << m.precision
                      << ",\"recall\":" << m.recall << ",\"f1\":" << m.f1_score
                      << ",\"tp\":" << m.true_positives << ",\"fp\":" << m.false_positives
                      << ",\"fn\":" << m.false_negatives << "}";
                }
                f << "]}";
                f.close();
                break;
            }
        } catch (...) {}
    }
}

void loadBestModelMetrics() {
    vector<string> paths;
    // Try current directory and executable location first
    try {
        paths.push_back((fs::current_path() / "best_model_metrics.json").string());
    } catch (...) {}
    paths.push_back("best_model_metrics.json");
    paths.push_back("./best_model_metrics.json");
    string model_dir = getModelDir();
    if (!model_dir.empty()) {
        paths.push_back((fs::path(model_dir) / "best_model_metrics.json").string());
    }
    paths.push_back("x64/Release/best_model_metrics.json");
    for (const auto& p : paths) {
        try {
            ifstream f(p);
            if (f.is_open()) {
                string content((istreambuf_iterator<char>(f)), istreambuf_iterator<char>());
                f.close();
                auto extractNum = [&content](const string& key) -> double {
                    size_t pos = content.find(key);
                    if (pos == string::npos) return 0;
                    pos += key.size();
                    size_t end = content.find_first_of(",}", pos);
                    if (end == string::npos) return 0;
                    try { return stod(content.substr(pos, end - pos)); } catch (...) { return 0; }
                };
                g_final_accuracy = extractNum("\"accuracy\":");
                g_final_macro_f1 = extractNum("\"macro_f1\":");
                g_display_num_classes = static_cast<int>(extractNum("\"num_classes\":"));
                g_final_class_metrics.clear();
                size_t arr_pos = content.find("\"metrics\":[");
                if (arr_pos != string::npos) {
                    arr_pos += 11;
                    auto extractFrom = [](const string& s, const string& key) -> double {
                        size_t p = s.find(key);
                        if (p == string::npos) return 0;
                        p += key.size();
                        size_t e = s.find_first_of(",}", p);
                        if (e == string::npos) return 0;
                        try { return stod(s.substr(p, e - p)); } catch (...) { return 0; }
                    };
                    while (arr_pos < content.size()) {
                        size_t obj_start = content.find('{', arr_pos);
                        if (obj_start == string::npos) break;
                        size_t obj_end = content.find('}', obj_start);
                        if (obj_end == string::npos) break;
                        string obj = content.substr(obj_start, obj_end - obj_start + 1);
                        NeuralNetwork::ClassMetrics m;
                        m.class_id = static_cast<int>(extractFrom(obj, "\"class_id\":"));
                        m.precision = extractFrom(obj, "\"precision\":");
                        m.recall = extractFrom(obj, "\"recall\":");
                        m.f1_score = extractFrom(obj, "\"f1\":");
                        m.true_positives = static_cast<int>(extractFrom(obj, "\"tp\":"));
                        m.false_positives = static_cast<int>(extractFrom(obj, "\"fp\":"));
                        m.false_negatives = static_cast<int>(extractFrom(obj, "\"fn\":"));
                        g_final_class_metrics.push_back(m);
                        arr_pos = obj_end + 1;
                        if (arr_pos >= content.size() || content[arr_pos] == ']') break;
                    }
                }
                break;
            }
        } catch (...) {}
    }
}

void saveTrainingHistory() {
    if (g_training_history.empty()) return;
    vector<string> paths;
    string model_dir = getModelDir();
    if (!model_dir.empty()) {
        paths.push_back((fs::path(model_dir) / "training_history.json").string());
    }
    paths.push_back("training_history.json");
    paths.push_back("x64/Release/training_history.json");
    for (const auto& p : paths) {
        try {
            ofstream f(p);
            if (f.is_open()) {
                f << fixed << setprecision(6);
                f << "[";
                for (size_t i = 0; i < g_training_history.size(); ++i) {
                    const auto& s = g_training_history[i];
                    if (i > 0) f << ",";
                    f << "{\"stage\":" << s.stage << ",\"epoch\":" << s.epoch
                      << ",\"accuracy\":" << s.accuracy << ",\"loss\":" << s.loss
                      << ",\"samples_processed\":" << s.samples_processed
                      << ",\"samples_accepted\":" << s.samples_accepted << "}";
                }
                f << "]";
                f.close();
                break;
            }
        } catch (...) {}
    }
}

void loadTrainingHistory() {
    vector<string> paths;
    string model_dir = getModelDir();
    if (!model_dir.empty()) {
        paths.push_back((fs::path(model_dir) / "training_history.json").string());
    }
    paths.push_back("training_history.json");
    paths.push_back("x64/Release/training_history.json");
    for (const auto& p : paths) {
        try {
            ifstream f(p);
            if (f.is_open()) {
                string content((istreambuf_iterator<char>(f)), istreambuf_iterator<char>());
                f.close();
                g_training_history.clear();
                auto extractFrom = [](const string& obj, const string& key) -> double {
                    size_t p = obj.find(key);
                    if (p == string::npos) return 0;
                    p += key.size();
                    size_t e = obj.find_first_of(",}]", p);
                    if (e == string::npos) return 0;
                    try { return stod(obj.substr(p, e - p)); } catch (...) { return 0; }
                };
                size_t pos = 0;
                while ((pos = content.find('{', pos)) != string::npos) {
                    size_t end = content.find('}', pos);
                    if (end == string::npos) break;
                    string obj = content.substr(pos, end - pos + 1);
                    TrainingStats s;
                    s.stage = static_cast<int>(extractFrom(obj, "\"stage\":"));
                    s.epoch = static_cast<int>(extractFrom(obj, "\"epoch\":"));
                    s.accuracy = extractFrom(obj, "\"accuracy\":");
                    s.loss = extractFrom(obj, "\"loss\":");
                    s.samples_processed = static_cast<int>(extractFrom(obj, "\"samples_processed\":"));
                    s.samples_accepted = static_cast<int>(extractFrom(obj, "\"samples_accepted\":"));
                    g_training_history.push_back(s);
                    pos = end + 1;
                }
                break;
            }
        } catch (...) {}
    }
}

// Выбор папки с обучающими данными
string openFolderDialog() {
#ifdef _WIN32
    BROWSEINFOA bi = {0};
    bi.ulFlags = BIF_RETURNONLYFSDIRS | BIF_NEWDIALOGSTYLE;
    bi.lpszTitle = "Select training folder";
    LPITEMIDLIST pidl = SHBrowseForFolderA(&bi);
    if (pidl != nullptr) {
        char path[MAX_PATH];
        if (SHGetPathFromIDListA(pidl, path)) {
            CoTaskMemFree(pidl);
            return string(path);
        }
        CoTaskMemFree(pidl);
    }
#endif
    return "";
}

void drawBPLA(Mat& img, Point position, Size size) {
    // Military BPLA (UAV) drawing in camouflage colors
    int width = size.width;
    int height = size.height;
    int x = position.x;
    int y = position.y;
    
    // Main body (fuselage) - dark green
    Scalar body_color(30, 60, 30);
    Point body_points[4];
    body_points[0] = Point(x + width/2 - 15, y + height/2);
    body_points[1] = Point(x + width/2 + 15, y + height/2);
    body_points[2] = Point(x + width/2 + 10, y + height - 10);
    body_points[3] = Point(x + width/2 - 10, y + height - 10);
    fillPoly(img, vector<vector<Point>>{vector<Point>(body_points, body_points + 4)}, body_color);
    
    // Wings - medium green
    Scalar wing_color(50, 80, 50);
    // Left wing
    Point left_wing[4];
    left_wing[0] = Point(x + width/2 - 15, y + height/2);
    left_wing[1] = Point(x + 5, y + height/2 + 5);
    left_wing[2] = Point(x + 5, y + height/2 + 15);
    left_wing[3] = Point(x + width/2 - 10, y + height/2 + 10);
    fillPoly(img, vector<vector<Point>>{vector<Point>(left_wing, left_wing + 4)}, wing_color);
    
    // Right wing
    Point right_wing[4];
    right_wing[0] = Point(x + width/2 + 15, y + height/2);
    right_wing[1] = Point(x + width - 5, y + height/2 + 5);
    right_wing[2] = Point(x + width - 5, y + height/2 + 15);
    right_wing[3] = Point(x + width/2 + 10, y + height/2 + 10);
    fillPoly(img, vector<vector<Point>>{vector<Point>(right_wing, right_wing + 4)}, wing_color);
    
    // Tail - dark green
    Point tail[3];
    tail[0] = Point(x + width/2, y + height - 10);
    tail[1] = Point(x + width/2 - 8, y + height - 25);
    tail[2] = Point(x + width/2 + 8, y + height - 25);
    fillPoly(img, vector<vector<Point>>{vector<Point>(tail, tail + 3)}, body_color);
    
    // Propeller (front) - accent color
    circle(img, Point(x + width/2, y + height/2 - 5), 8, ACCENT_COLOR, 2);
    line(img, Point(x + width/2 - 8, y + height/2 - 5), Point(x + width/2 + 8, y + height/2 - 5), ACCENT_COLOR, 2);
    line(img, Point(x + width/2, y + height/2 - 13), Point(x + width/2, y + height/2 + 3), ACCENT_COLOR, 2);
    
    // Outline in military green
    Scalar outline_color(100, 150, 100);
    polylines(img, vector<vector<Point>>{vector<Point>(body_points, body_points + 4)}, true, outline_color, 1);
    polylines(img, vector<vector<Point>>{vector<Point>(left_wing, left_wing + 4)}, true, outline_color, 1);
    polylines(img, vector<vector<Point>>{vector<Point>(right_wing, right_wing + 4)}, true, outline_color, 1);
    polylines(img, vector<vector<Point>>{vector<Point>(tail, tail + 3)}, true, outline_color, 1);
}

// Отрисовка графика обучения с двумя осями Y (Loss слева, Accuracy справа)
void drawTrainingChart(Mat& img, Rect chart_rect, const vector<TrainingStats>& history) {
    if (history.empty()) {
        putText(img, "No training data available", 
                Point(chart_rect.x + 10, chart_rect.y + chart_rect.height / 2),
                FONT_HERSHEY_SIMPLEX, 0.6, Scalar(150, 150, 150), 1, LINE_AA);
        return;
    }
    
    // Фон графика
    rectangle(img, chart_rect, Scalar(15, 25, 15), -1);
    rectangle(img, chart_rect, Scalar(80, 80, 80), 2);
    
    // Область для графика (с отступами для двух осей Y)
    int padding_left = 55;
    int padding_right = 55;
    int padding_top = 35;
    int padding_bottom = 45;
    int chart_x = chart_rect.x + padding_left;
    int chart_y = chart_rect.y + padding_top;
    int chart_width = chart_rect.width - padding_left - padding_right;
    int chart_height = chart_rect.height - padding_top - padding_bottom;
    
    // Находим максимальные значения
    double max_loss = 0.0;
    for (const auto& stats : history) {
        if (stats.loss > max_loss) max_loss = stats.loss;
    }
    max_loss = max(max_loss * 1.1, 0.01);
    
    // Оси: X внизу, Y слева (Loss), Y справа (Accuracy)
    Point origin(chart_x, chart_y + chart_height);
    Point x_end(chart_x + chart_width, chart_y + chart_height);
    Point y_left_end(chart_x, chart_y);
    Point y_right_end(chart_x + chart_width, chart_y);
    
    line(img, origin, x_end, Scalar(100, 100, 100), 2, LINE_AA);
    line(img, origin, y_left_end, Scalar(100, 100, 100), 2, LINE_AA);
    line(img, Point(chart_x + chart_width, chart_y + chart_height), y_right_end, Scalar(100, 100, 100), 2, LINE_AA);
    
    // Подпись оси X
    putText(img, "Epoch", Point(chart_x + chart_width / 2 - 25, chart_y + chart_height + 38),
            FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1, LINE_AA);
    
    // Левая ось Y — Loss (красная)
    putText(img, "Loss", Point(chart_rect.x + 5, chart_y - 5),
            FONT_HERSHEY_SIMPLEX, 0.5, Scalar(100, 100, 255), 1, LINE_AA);
    stringstream ss;
    ss << fixed << setprecision(1) << max_loss;
    putText(img, ss.str(), Point(chart_x - 45, chart_y + 5), FONT_HERSHEY_SIMPLEX, 0.4, Scalar(150, 150, 255), 1, LINE_AA);
    putText(img, "0", Point(chart_x - 20, chart_y + chart_height + 5), FONT_HERSHEY_SIMPLEX, 0.4, Scalar(150, 150, 255), 1, LINE_AA);
    
    // Правая ось Y — Accuracy (зелёная)
    putText(img, "Acc", Point(chart_x + chart_width - 25, chart_y - 5),
            FONT_HERSHEY_SIMPLEX, 0.5, Scalar(100, 255, 100), 1, LINE_AA);
    putText(img, "1.0", Point(chart_x + chart_width + 5, chart_y + 5), FONT_HERSHEY_SIMPLEX, 0.4, Scalar(150, 255, 150), 1, LINE_AA);
    putText(img, "0", Point(chart_x + chart_width + 5, chart_y + chart_height + 5), FONT_HERSHEY_SIMPLEX, 0.4, Scalar(150, 255, 150), 1, LINE_AA);
    
    double loss_scale = chart_height / max_loss;
    double accuracy_scale = chart_height;  // 0..1 -> 0..chart_height
    
    // Отрисовка линий
    if (history.size() > 1) {
        for (size_t i = 0; i < history.size() - 1; ++i) {
            double t1 = (double)i / (history.size() - 1);
            double t2 = (double)(i + 1) / (history.size() - 1);
            int x1 = chart_x + static_cast<int>(t1 * chart_width);
            int x2 = chart_x + static_cast<int>(t2 * chart_width);
            
            // Loss (левая шкала)
            int ly1 = chart_y + chart_height - static_cast<int>(min(history[i].loss, max_loss) * loss_scale);
            int ly2 = chart_y + chart_height - static_cast<int>(min(history[i + 1].loss, max_loss) * loss_scale);
            line(img, Point(x1, ly1), Point(x2, ly2), Scalar(0, 0, 255), 2, LINE_AA);
            
            // Accuracy (правая шкала, 0-1)
            int ay1 = chart_y + chart_height - static_cast<int>(history[i].accuracy * accuracy_scale);
            int ay2 = chart_y + chart_height - static_cast<int>(history[i + 1].accuracy * accuracy_scale);
            line(img, Point(x1, ay1), Point(x2, ay2), Scalar(0, 255, 0), 2, LINE_AA);
        }
    }
    
    // Легенда
    int legend_x = chart_x + chart_width - 160;
    int legend_y = chart_y + 5;
    line(img, Point(legend_x, legend_y), Point(legend_x + 25, legend_y), Scalar(0, 0, 255), 2, LINE_AA);
    putText(img, "Loss", Point(legend_x + 30, legend_y + 5), FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1, LINE_AA);
    line(img, Point(legend_x, legend_y + 22), Point(legend_x + 25, legend_y + 22), Scalar(0, 255, 0), 2, LINE_AA);
    putText(img, "Accuracy", Point(legend_x + 30, legend_y + 27), FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1, LINE_AA);
    
    putText(img, "Training Progress", Point(chart_x + chart_width / 2 - 65, chart_rect.y + 18),
            FONT_HERSHEY_SIMPLEX, 0.65, ACCENT_COLOR, 1, LINE_AA);
}

void drawImageWithResult(Mat& img, int start_x, int start_y, int width, int height) {
    Rect img_area(start_x, start_y, width, height);
    rectangle(img, img_area, Scalar(50, 50, 50), 2);
    
    if (!g_current_image.empty()) {
        Mat display;
        double scale = min((double)width / g_current_image.cols, (double)height / g_current_image.rows);
        resize(g_current_image, display, Size(), scale, scale);
        
        int dx = (width - display.cols) / 2;
        int dy = (height - display.rows) / 2;
        display.copyTo(img(Rect(start_x + dx, start_y + dy, display.cols, display.rows)));
    } else {
        string msg = "No image loaded";
        Size textSize = getTextSize(msg, FONT_HERSHEY_SIMPLEX, 0.8, 2, 0);
        putText(img, msg, 
                Point(start_x + (width - textSize.width) / 2, start_y + height / 2), 
                FONT_HERSHEY_SIMPLEX, 0.8, Scalar(100, 100, 100), 2, LINE_AA);
    }
}

void drawResultAndParams(Mat& img, int start_x, int start_y) {
    if (!g_classification_result.empty()) {
        string display_name = g_classification_result;
        bool has_real_name = (display_name.find("Class ") != 0);
        string result_text;
        if (has_real_name && g_predicted_class_id >= 0) {
            result_text = "Result: " + display_name + " (Class " + to_string(g_predicted_class_id) + ")";
        } else {
            result_text = "Result: " + display_name;
        }
        putText(img, result_text, Point(start_x, start_y), FONT_HERSHEY_SIMPLEX, 0.9, TEXT_COLOR, 2, LINE_AA);
        
        int text_y = start_y + 35;
        stringstream ss;
        ss << fixed << setprecision(1) << (g_classification_confidence * 100) << "%";
        putText(img, "Confidence: " + ss.str(), Point(start_x, text_y), FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2, LINE_AA);
        
        if (!has_real_name && g_predicted_class_id >= 0) {
            putText(img, "Class names not loaded. Set training folder (Training -> Select Folder) and reload image, or retrain with folders named by class (helicopter, aircraft, etc.).", 
                    Point(start_x, text_y + 35), FONT_HERSHEY_SIMPLEX, 0.45, WARNING_COLOR, 1, LINE_AA);
        }
    }
}

void drawClassStatisticsTable(Mat& img, int start_x, int start_y, int table_width_param) {
    int table_x = start_x;
    int table_width = (table_width_param > 0) ? table_width_param : (WINDOW_WIDTH - start_x - 50);
    int row_height = 70;
    int header_height = 40;
    
    const vector<NeuralNetwork::ClassMetrics>& display_metrics = 
        g_class_metrics.empty() ? g_final_class_metrics : g_class_metrics;
    double display_macro_f1 = g_class_metrics.empty() ? g_final_macro_f1 : g_macro_f1;
    
    int title_offset = 25;
    putText(img, "Best model metrics:", Point(table_x, start_y + title_offset), FONT_HERSHEY_SIMPLEX, 0.8, TEXT_COLOR, 2, LINE_AA);
    
    int table_start_y = start_y + title_offset + 35;
    
    // Колонки: Class, Precision, Recall, F1-score, TP, FP, FN (без Note)
    const int PAD = 8;
    vector<string> headers = {"Class", "Precision", "Recall", "F1-score", "TP", "FP", "FN"};
    const vector<int> col_ratios = {70, 75, 75, 75, 40, 40, 40};
    const int ratio_sum = 70+75+75+75+40+40+40;
    vector<int> col_widths(7);
    for (int i = 0; i < 7; ++i) {
        col_widths[i] = max(45, static_cast<int>(col_ratios[i] * (double)table_width / ratio_sum));
    }
    int sum = 0;
    for (int i = 0; i < 7; ++i) sum += col_widths[i];
    col_widths[6] += table_width - sum;
    
    Rect header_rect(table_x, table_start_y, table_width, header_height);
    rectangle(img, header_rect, TABLE_HEADER_COLOR, -1);
    rectangle(img, header_rect, Scalar(100, 100, 100), 1);
    
    Scalar cell_border(80, 100, 80);
    int col_x = table_x;
    for (size_t i = 0; i < headers.size(); ++i) {
        putText(img, headers[i], Point(col_x + PAD, table_start_y + 28), FONT_HERSHEY_SIMPLEX, 0.55, TEXT_COLOR, 1, LINE_AA);
        col_x += col_widths[i];
        if (i < headers.size() - 1) {
            line(img, Point(col_x, table_start_y), Point(col_x, table_start_y + header_height), cell_border, 1, LINE_AA);
        }
    }
    line(img, Point(table_x + table_width, table_start_y), Point(table_x + table_width, table_start_y + header_height), cell_border, 1, LINE_AA);
    
    int current_y = table_start_y + header_height;
    
    if (!display_metrics.empty()) {
        for (size_t i = 0; i < display_metrics.size(); ++i) {
            const auto& metric = display_metrics[i];
            
            Scalar row_color = (i % 2 == 0) ? TABLE_ROW_COLOR1 : TABLE_ROW_COLOR2;
            Rect row_rect(table_x, current_y, table_width, row_height);
            rectangle(img, row_rect, row_color, -1);
            rectangle(img, row_rect, Scalar(60, 60, 60), 1);
            
            string class_name = g_classifier ? g_classifier->getClassName(metric.class_id) : "Class " + to_string(metric.class_id);
            int cell_x = table_x;
            int text_y = current_y + row_height / 2 + 10;
            putText(img, class_name, Point(cell_x + PAD, text_y), FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1, LINE_AA);
            cell_x += col_widths[0];
            
            stringstream ss;
            ss << fixed << setprecision(3) << (metric.precision * 100) << "%";
            putText(img, ss.str(), Point(cell_x + PAD, text_y), FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1, LINE_AA);
            cell_x += col_widths[1];
            
            ss.str("");
            ss << fixed << setprecision(3) << (metric.recall * 100) << "%";
            putText(img, ss.str(), Point(cell_x + PAD, text_y), FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1, LINE_AA);
            cell_x += col_widths[2];
            
            ss.str("");
            ss << fixed << setprecision(3) << (metric.f1_score * 100) << "%";
            Scalar f1_color = (metric.f1_score < 0.5) ? WARNING_COLOR : TEXT_COLOR;
            putText(img, ss.str(), Point(cell_x + PAD, text_y), FONT_HERSHEY_SIMPLEX, 0.5, f1_color, 1, LINE_AA);
            cell_x += col_widths[3];
            
            putText(img, to_string(metric.true_positives), Point(cell_x + PAD, text_y), FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1, LINE_AA);
            cell_x += col_widths[4];
            
            putText(img, to_string(metric.false_positives), Point(cell_x + PAD, text_y), FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1, LINE_AA);
            cell_x += col_widths[5];
            
            putText(img, to_string(metric.false_negatives), Point(cell_x + PAD, text_y), FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1, LINE_AA);
            
            current_y += row_height;
        }
    } else {
        putText(img, "No metrics. Train the model first.", Point(table_x + PAD, current_y + row_height / 2 + 10), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(150, 150, 150), 1, LINE_AA);
        current_y += row_height;
    }
    
    int table_bottom_y = current_y;
    int vx = table_x;
    for (size_t c = 0; c < col_widths.size(); ++c) {
        vx += col_widths[c];
        line(img, Point(vx, table_start_y), Point(vx, table_bottom_y), cell_border, 1, LINE_AA);
    }
    
    current_y += 20;
    stringstream ss;
    ss << fixed << setprecision(3) << (display_macro_f1 * 100) << "%";
    putText(img, "Macro-averaged F1-score: " + ss.str(), Point(table_x, current_y), FONT_HERSHEY_SIMPLEX, 0.7, ACCENT_COLOR, 2, LINE_AA);
    
    bool has_low_f1 = false;
    for (const auto& m : display_metrics) {
        if (m.f1_score < 0.5) { has_low_f1 = true; break; }
    }
    if (has_low_f1) {
        current_y += 35;
        putText(img, "WARNING: Low recognition quality in some classes", Point(table_x, current_y), FONT_HERSHEY_SIMPLEX, 0.6, WARNING_COLOR, 1, LINE_AA);
    }
}

void drawMainScreen() {
    lock_guard<mutex> lock(g_ui_mutex);
    g_main_ui = Mat(WINDOW_HEIGHT, WINDOW_WIDTH, CV_8UC3, BG_COLOR);
    
    const int left_width = static_cast<int>(WINDOW_WIDTH * 0.55);
    const int right_x = left_width + 20;
    const int right_width = WINDOW_WIDTH - right_x - 30;
    const int right_y = 120;
    const int right_height = WINDOW_HEIGHT - right_y - 50;
    
    drawBPLA(g_main_ui, Point(WINDOW_WIDTH - 200, 30), Size(150, 80));
    putText(g_main_ui, "Image Classifier", Point(50, 50), FONT_HERSHEY_SIMPLEX, 1.2, TEXT_COLOR, 2, LINE_AA);
    
    int left_x = 50;
    int left_content_width = left_width - 60;
    
    int img_y = 120;
    int img_height = 320;
    drawImageWithResult(g_main_ui, left_x, img_y, left_content_width, img_height);
    
    g_button_load_photo = Rect(left_x, img_y + img_height + 5, left_content_width, 65);
    bool hover_load = g_button_load_photo.contains(g_mouse_pos);
    drawButton(g_main_ui, g_button_load_photo, "Load Photo", hover_load);
    
    int result_y = img_y + img_height + 5 + 65 + 45;
    drawResultAndParams(g_main_ui, left_x, result_y);
    
    int chart_y = img_y + img_height + 5 + 65 + 165;
    int chart_height = 250;
    if (!g_training_history.empty() && chart_y + chart_height < right_y + right_height - 95) {
        Rect chart_rect(left_x, chart_y, left_content_width, chart_height);
        drawTrainingChart(g_main_ui, chart_rect, g_training_history);
    }
    
    int training_btn_y = right_y + right_height - 75;
    g_button_training = Rect(left_x, training_btn_y, left_content_width, 75);
    bool hover_training = g_button_training.contains(g_mouse_pos);
    drawButton(g_main_ui, g_button_training, "Training", hover_training);
    
    Rect right_panel(right_x, right_y, right_width, right_height);
    rectangle(g_main_ui, right_panel, Scalar(25, 40, 25), -1);
    rectangle(g_main_ui, right_panel, Scalar(80, 100, 80), 2);
    
    const vector<NeuralNetwork::ClassMetrics>& disp_m = 
        g_class_metrics.empty() ? g_final_class_metrics : g_class_metrics;
    int table_rows = disp_m.empty() ? 1 : static_cast<int>(disp_m.size());
    bool has_warning = false;
    for (const auto& m : disp_m) { if (m.f1_score < 0.5) { has_warning = true; break; } }
    int metrics_content_height = 60 + 40 + table_rows * 70 + 20 + 35 + (has_warning ? 55 : 0);
    
    if (g_training_completed && !g_training_history.empty()) {
        putText(g_main_ui, "Training complete", Point(right_x + 15, right_y + 25), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2, LINE_AA);
        stringstream ss;
        ss << fixed << setprecision(1) << (g_final_accuracy * 100) << "%";
        putText(g_main_ui, ss.str(), Point(right_x + 15, right_y + 50), FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 1, LINE_AA);
        metrics_content_height += 50;
    }
    
    int metrics_inner_y = right_y + (g_training_completed ? 70 : 10);
    int metrics_inner_height = right_height - (g_training_completed ? 70 : 10);
    
    if (metrics_content_height > metrics_inner_height) {
        g_metrics_scroll_offset = max(0, min(g_metrics_scroll_offset, metrics_content_height - metrics_inner_height));
        Mat metrics_canvas(metrics_content_height, right_width - 20, CV_8UC3, Scalar(25, 40, 25));
        drawClassStatisticsTable(metrics_canvas, 5, 10, right_width - 35);
        int src_y = g_metrics_scroll_offset;
        int copy_h = min(metrics_inner_height, metrics_content_height - src_y);
        if (copy_h > 0) {
            Rect src_rect(0, src_y, right_width - 20, copy_h);
            Rect dst_rect(right_x + 10, metrics_inner_y, right_width - 20, copy_h);
            metrics_canvas(src_rect).copyTo(g_main_ui(dst_rect));
        }
        putText(g_main_ui, "Scroll: wheel", Point(right_x + right_width - 90, right_y + right_height - 5), FONT_HERSHEY_SIMPLEX, 0.4, Scalar(120, 120, 120), 1, LINE_AA);
    } else {
        g_metrics_scroll_offset = 0;
        drawClassStatisticsTable(g_main_ui, right_x + 5, metrics_inner_y, right_width - 35);
    }
    
    putText(g_main_ui, "Status: " + g_status_text, 
            Point(50, WINDOW_HEIGHT - 25), 
            FONT_HERSHEY_SIMPLEX, 0.5, Scalar(150, 150, 150), 1, LINE_AA);
    
    string footer = "by Artem Vladimirovich Tretyakov & Daniil Alexandrovich Molchanov";
    int baseline = 0;
    Size footerSize = getTextSize(footer, FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseline);
    Point footerPos(WINDOW_WIDTH - footerSize.width - 20,
                    WINDOW_HEIGHT - 20);
    putText(g_main_ui, footer,
            footerPos,
            FONT_HERSHEY_SIMPLEX, 0.4, Scalar(80, 160, 80), 1, LINE_AA);

    drawNotifications(g_main_ui);

    imshow("Image Classifier", g_main_ui);
#ifdef _WIN32
    disableWindowClose("Image Classifier");
#endif
}

void drawTrainingScreen() {
    lock_guard<mutex> lock(g_ui_mutex);
    g_main_ui = Mat(WINDOW_HEIGHT, WINDOW_WIDTH, CV_8UC3, BG_COLOR);
    
    drawBPLA(g_main_ui, Point(WINDOW_WIDTH - 200, 30), Size(150, 80));
    
    bool training_done = g_training_completed && !g_training_active;
    
    Rect status_banner(50, 50, WINDOW_WIDTH - 100, 55);
    if (training_done) {
        auto ms_done = chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now().time_since_epoch()).count();
        bool pulse_green = (ms_done / 400) % 2 == 0;
        Scalar green_color = pulse_green ? Scalar(0, 255, 0) : Scalar(0, 200, 0);  // BGR: blinking green
        rectangle(g_main_ui, status_banner, green_color, -1);
        rectangle(g_main_ui, status_banner, Scalar(0, 255, 0), 2);
        putText(g_main_ui, "Complete training",
                Point(status_banner.x + (status_banner.width - 160) / 2, status_banner.y + 38),
                FONT_HERSHEY_SIMPLEX, 0.95, Scalar(0, 0, 0), 2, LINE_AA);
    } else if (g_training_active) {
        bool is_preprocessing = (g_training_status == "Preprocessing..." ||
                                 g_training_status.find("Preparing") != string::npos ||
                                 g_training_status.find("Starting") != string::npos ||
                                 g_training_status.find("Loading") != string::npos);
        bool is_saving = (g_training_status.find("Saving") != string::npos);
        auto ms = chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now().time_since_epoch()).count();
        bool pulse = (ms / 400) % 2 == 0;
        if (is_preprocessing) {
            Scalar blue_color = pulse ? Scalar(220, 160, 0) : Scalar(180, 120, 0);  // BGR: blue
            rectangle(g_main_ui, status_banner, blue_color, -1);
            rectangle(g_main_ui, status_banner, Scalar(255, 200, 0), 2);
            putText(g_main_ui, "Preprocessing",
                    Point(status_banner.x + (status_banner.width - 120) / 2, status_banner.y + 38),
                    FONT_HERSHEY_SIMPLEX, 0.9, Scalar(200, 200, 255), 2, LINE_AA);
        } else if (is_saving) {
            Scalar yellow_color = pulse ? Scalar(0, 255, 255) : Scalar(0, 200, 200);  // BGR: yellow
            rectangle(g_main_ui, status_banner, yellow_color, -1);
            rectangle(g_main_ui, status_banner, Scalar(0, 255, 255), 2);
            putText(g_main_ui, "Saving",
                    Point(status_banner.x + (status_banner.width - 60) / 2, status_banner.y + 38),
                    FONT_HERSHEY_SIMPLEX, 0.9, Scalar(0, 0, 0), 2, LINE_AA);
        } else {
            Scalar red_color = pulse ? Scalar(0, 0, 220) : Scalar(0, 0, 160);
            rectangle(g_main_ui, status_banner, red_color, -1);
            rectangle(g_main_ui, status_banner, Scalar(0, 0, 255), 2);
            putText(g_main_ui, "Training",
                    Point(status_banner.x + (status_banner.width - 80) / 2, status_banner.y + 38),
                    FONT_HERSHEY_SIMPLEX, 0.9, Scalar(200, 200, 255), 2, LINE_AA);
        }
    } else {
        rectangle(g_main_ui, status_banner, Scalar(40, 50, 40), -1);
        rectangle(g_main_ui, status_banner, Scalar(80, 100, 80), 2);
        putText(g_main_ui, "Ready to train",
                Point(status_banner.x + (status_banner.width - 120) / 2, status_banner.y + 38),
                FONT_HERSHEY_SIMPLEX, 0.9, TEXT_COLOR, 2, LINE_AA);
    }
    
    int y_pos = 130;
    
    double circle1_progress = 0.0;
    double circle2_progress = 0.0;
    bool circle1_complete = false;
    bool circle2_complete = false;
    if (g_total_epochs > 0) {
        if (g_current_stage >= 3) {
            circle1_progress = 1.0;
            circle1_complete = true;
            circle2_progress = (g_current_stage >= 4) ? 1.0 : min(1.0, g_current_epoch / static_cast<double>(g_stage3_epochs));
            circle2_complete = (g_current_stage >= 4);
        } else if (g_current_stage == 2) {
            circle1_progress = min(1.0, g_current_epoch / static_cast<double>(g_total_epochs));
            circle1_complete = (g_current_epoch >= g_total_epochs);
        }
    }
    if (training_done) {
        circle1_complete = true;
        circle2_complete = true;
        circle1_progress = 1.0;
        circle2_progress = 1.0;
    }
    
    int circle_radius = 130;
    int circle1_x = WINDOW_WIDTH / 4;
    int circle2_x = (3 * WINDOW_WIDTH) / 4;
    int circle_y = y_pos + circle_radius + 30;
    
    drawCircleProgress(g_main_ui, Point(circle1_x, circle_y), circle_radius, circle1_progress, circle1_complete, "Cycle 1");
    drawCircleProgress(g_main_ui, Point(circle2_x, circle_y), circle_radius, circle2_progress, circle2_complete, "Cycle 2");
    
    int center_x = WINDOW_WIDTH / 2;
    int info_y = circle_y - 60;
    auto drawCenteredInfo = [&](const string& text, int& y) {
        int bl = 0;
        Size ts = getTextSize(text, FONT_HERSHEY_SIMPLEX, 0.6, 1, &bl);
        putText(g_main_ui, text, Point(center_x - ts.width / 2, y), FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 1, LINE_AA);
        y += 28;
    };
    if (g_training_active) {
        auto now = chrono::steady_clock::now();
        auto elapsed = now - g_training_start_time;
        drawCenteredInfo("Time: " + formatTime(elapsed), info_y);
        stringstream ss;
        ss << "Batch Size: " << g_batch_size << " | LR: " << fixed << setprecision(4) << g_learning_rate;
        drawCenteredInfo(ss.str(), info_y);
        if (g_current_epoch > 0) {
            ss.str("");
            ss << "Loss: " << fixed << setprecision(4) << g_current_loss << " | Acc: " << setprecision(2) << (g_current_accuracy * 100) << "%";
            drawCenteredInfo(ss.str(), info_y);
        }
        if (g_total_images > 0) {
            ss.str("");
            ss << "Epoch " << g_current_epoch << "/" << g_total_epochs;
            drawCenteredInfo(ss.str(), info_y);
            if (g_processed_images > 0) {
                ss.str("");
                ss << "Samples: " << g_processed_images << " (from " << g_total_images << " images)";
                drawCenteredInfo(ss.str(), info_y);
            }
        }
        if (g_current_epoch > 0) {
            auto now = chrono::steady_clock::now();
            auto elapsed = now - g_training_start_time;
            double elapsed_sec = static_cast<double>(chrono::duration_cast<chrono::seconds>(elapsed).count());
            double avg_per_epoch = elapsed_sec / g_current_epoch;
            double remaining = avg_per_epoch * max(0, g_total_epochs - g_current_epoch);
            string eta_str = formatTime(chrono::seconds(static_cast<long long>(remaining)));
            drawCenteredInfo("ETA: " + eta_str, info_y);
        }
        drawCenteredInfo(g_fast_mode ? "Mode: FAST" : "Mode: FULL", info_y);
    }
    
    if (circle1_complete && !circle2_complete) {
        string tc = "Training complete";
        int tcb = 0;
        Size tcs = getTextSize(tc, FONT_HERSHEY_SIMPLEX, 0.55, 1, &tcb);
        putText(g_main_ui, tc, Point(circle1_x - tcs.width/2, circle_y + circle_radius + 55), FONT_HERSHEY_SIMPLEX, 0.55, Scalar(0, 255, 0), 1, LINE_AA);
    }
    
    y_pos = circle_y + circle_radius + 90;
    
    putText(g_main_ui, "Dataset: " + g_training_dir, Point(50, y_pos), FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 1, LINE_AA);
    y_pos += 30;
    
    string accel = CudaAccelerator::isAvailable() ? "CUDA" : "CPU";
    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    {
        stringstream ss;
        ss << "Accel: " << accel << " | Threads: " << threads;
        putText(g_main_ui, ss.str(), Point(50, y_pos), FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 1, LINE_AA);
    }
    y_pos += 35;
    
    putText(g_main_ui, g_training_status, Point(50, y_pos), FONT_HERSHEY_SIMPLEX, 0.8, TEXT_COLOR, 2, LINE_AA);
    y_pos += 45;
    
    if (g_total_epochs > 0) {
        stringstream ss;
        if (g_current_stage == 2) {
            ss << "Epoch: " << g_current_epoch << " / " << g_total_epochs << " (Cycle 1)";
        } else if (g_current_stage == 3) {
            ss << "Epoch: " << g_current_epoch << " / " << g_stage3_epochs << " (Cycle 2)";
        } else if (g_current_stage == 4) {
            ss << "Problem classes: " << g_stage4_epoch_count << " / " << g_stage4_total_epochs;
        } else {
            ss << "Epoch: " << g_current_epoch << " / " << g_total_epochs;
        }
        putText(g_main_ui, ss.str(), Point(50, y_pos), FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2, LINE_AA);
        y_pos += 45;
        drawProgressBar(g_main_ui, Rect(50, y_pos, WINDOW_WIDTH - 100, 35), g_train_progress);
        
        if (g_current_stage == 4 || (g_training_completed && !g_problem_classes.empty())) {
            y_pos += 35 + 20;
            putText(g_main_ui, "Problematic classes", Point(50, y_pos), FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2, LINE_AA);
            y_pos += 25;
            drawProgressBar(g_main_ui, Rect(50, y_pos, WINDOW_WIDTH - 100, 35), g_problem_class_progress);
        }
        if (g_quality_control_progress > 0) {
            y_pos += 35 + 20;
            putText(g_main_ui, "Quality control", Point(50, y_pos), FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2, LINE_AA);
            y_pos += 25;
            drawProgressBar(g_main_ui, Rect(50, y_pos, WINDOW_WIDTH - 100, 35), g_quality_control_progress);
        }
    }
    
    bool hover_back = g_button_back.contains(g_mouse_pos);
    bool hover_start = g_button_start_training.contains(g_mouse_pos);
    bool hover_dir = g_button_select_dir.contains(g_mouse_pos);
    bool hover_mode = g_button_mode.contains(g_mouse_pos);
    drawButton(g_main_ui, g_button_back, "Back", hover_back);
    drawButton(g_main_ui, g_button_start_training, g_training_active ? "Training..." : "Start Training", hover_start);
    drawButton(g_main_ui, g_button_select_dir, "Select Folder", hover_dir);
    drawButton(g_main_ui, g_button_mode, g_fast_mode ? "Mode: FAST" : "Mode: FULL", hover_mode);
    
    y_pos += 80;
    
    // Additional training info (ETA and Mode moved to center between circles)
    y_pos += 20;
    
    // Если обучение завершено, показываем финальные результаты
    if (g_training_completed && !g_training_history.empty()) {
        y_pos += 30;
        
        // Финальные метрики
        putText(g_main_ui, "=== Training Completed ===", 
                Point(50, y_pos), 
                FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2, LINE_AA);
        y_pos += 40;
        
        stringstream ss;
        ss << fixed << setprecision(2);
        ss << "Overall Accuracy: " << (g_final_accuracy * 100) << "%";
        putText(g_main_ui, ss.str(), 
                Point(50, y_pos), 
                FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 1, LINE_AA);
        y_pos += 30;
        
        ss.str("");
        ss << "Macro F1-Score: " << (g_final_macro_f1 * 100) << "%";
        putText(g_main_ui, ss.str(), 
                Point(50, y_pos), 
                FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 1, LINE_AA);
        y_pos += 30;
        
        ss.str("");
        ss << "Weighted F1-Score: " << (g_final_weighted_f1 * 100) << "%";
        putText(g_main_ui, ss.str(), 
                Point(50, y_pos), 
                FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 1, LINE_AA);
        y_pos += 40;
        
        // Проблемные классы
        if (!g_problem_classes.empty()) {
            ss.str("");
            ss << "Problem Classes: [";
            for (size_t i = 0; i < g_problem_classes.size(); ++i) {
                ss << g_problem_classes[i];
                if (i < g_problem_classes.size() - 1) ss << ", ";
            }
            ss << "]";
            putText(g_main_ui, ss.str(), 
                    Point(50, y_pos), 
                    FONT_HERSHEY_SIMPLEX, 0.6, WARNING_COLOR, 1, LINE_AA);
            y_pos += 30;
        }
        
        y_pos += 20;
        
        // Class metrics (top problem classes)
        if (!g_final_class_metrics.empty() && y_pos < WINDOW_HEIGHT - 100) {
            y_pos += 20;
            putText(g_main_ui, "=== Class Metrics (Top Problem Classes) ===", 
                    Point(50, y_pos), 
                    FONT_HERSHEY_SIMPLEX, 0.6, ACCENT_COLOR, 1, LINE_AA);
            y_pos += 30;
            
            // Показываем только проблемные классы или топ-5 классов с низким F1
            vector<pair<int, double>> class_f1_scores;
            for (const auto& metric : g_final_class_metrics) {
                bool has_data = (metric.true_positives + metric.false_negatives > 0) || 
                               (metric.true_positives + metric.false_positives > 0);
                if (has_data) {
                    class_f1_scores.push_back({metric.class_id, metric.f1_score});
                }
            }
            
            // Сортируем по F1-score (от меньшего к большему)
            sort(class_f1_scores.begin(), class_f1_scores.end(),
                 [](const pair<int, double>& a, const pair<int, double>& b) {
                     return a.second < b.second;
                 });
            
            // Показываем топ-5 проблемных классов
            int show_count = min(5, static_cast<int>(class_f1_scores.size()));
            for (int i = 0; i < show_count && y_pos < WINDOW_HEIGHT - 100; ++i) {
                int class_id = class_f1_scores[i].first;
                auto it = find_if(g_final_class_metrics.begin(), g_final_class_metrics.end(),
                                 [class_id](const NeuralNetwork::ClassMetrics& m) {
                                     return m.class_id == class_id;
                                 });
                
                if (it != g_final_class_metrics.end()) {
                    string class_name = g_classifier ? g_classifier->getClassName(class_id) : 
                                       "Class " + to_string(class_id);
                    ss.str("");
                    ss << fixed << setprecision(2);
                    ss << class_name << ": P=" << (it->precision * 100) << "%, "
                       << "R=" << (it->recall * 100) << "%, "
                       << "F1=" << (it->f1_score * 100) << "%";
                    putText(g_main_ui, ss.str(), 
                            Point(50, y_pos), 
                            FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1, LINE_AA);
                    y_pos += 25;
                }
            }
        }
    } else if (!g_training_status.empty() && g_training_status.find("Finished") != string::npos) {
        putText(g_main_ui, "Training completed successfully!", 
                Point(50, y_pos), 
                FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0), 1, LINE_AA);
    }
    
    // Footer with authors
    string footer = "by Artem Vladimirovich Tretyakov & Daniil Alexandrovich Molchanov";
    int baseline = 0;
    Size footerSize = getTextSize(footer, FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseline);
    Point footerPos(WINDOW_WIDTH - footerSize.width - 20,
                    WINDOW_HEIGHT - 20);
    putText(g_main_ui, footer,
            footerPos,
            FONT_HERSHEY_SIMPLEX, 0.4, Scalar(80, 160, 80), 1, LINE_AA);

    drawNotifications(g_main_ui);

    imshow("Image Classifier", g_main_ui);
#ifdef _WIN32
    disableWindowClose("Image Classifier");
#endif
}

void onMouse(int event, int x, int y, int flags, void* userdata) {
    g_mouse_pos = Point(x, y);
    
#ifdef _WIN32
    if (event == EVENT_MOUSEWHEEL && g_app_state == STATE_MAIN) {
        int right_x = static_cast<int>(WINDOW_WIDTH * 0.55) + 20;
        int right_y = 120;
        int right_width = WINDOW_WIDTH - right_x - 30;
        int right_height = WINDOW_HEIGHT - 120 - 50;
        if (x >= right_x && x < right_x + right_width && y >= right_y && y < right_y + right_height) {
            int delta = getMouseWheelDelta(flags);
            g_metrics_scroll_offset -= delta * 40;
            g_metrics_scroll_offset = max(0, g_metrics_scroll_offset);
        }
    }
#endif

    if (event == EVENT_LBUTTONDOWN) {
        {
            lock_guard<mutex> lock(g_notifications_mutex);
            for (auto& n : g_notifications) {
                if (!n.closed && n.close_rect.contains(Point(x, y))) {
                    n.closed = true;
                    break;
                }
            }
        }
        removeClosedNotifications();
        if (g_app_state == STATE_MAIN) {
            // Load photo button
            if (g_button_load_photo.contains(Point(x, y))) {
                string path = openFileDialog();
                if (!path.empty()) {
                    loadImageAndClassify(path);
                }
            }
            // Training button
            else if (g_button_training.contains(Point(x, y))) {
                g_app_state = STATE_TRAINING;
                g_training_status = "Ready to train";
                g_training_completed = false;
            }
        } else if (g_app_state == STATE_TRAINING) {
            // Back button
            if (g_button_back.contains(Point(x, y))) {
                g_app_state = STATE_MAIN;
                if (g_training_completed && !g_current_image.empty()) {
                    reclassifyCurrentImage();
                }
            }
            // Start training button
            else if (g_button_start_training.contains(Point(x, y))) {
                if (!g_training_active) {
                    if (!fs::exists(g_training_dir)) {
                        pushNotification("Training folder does not exist: " + g_training_dir);
                    } else {
                        g_training_completed = false;
                        startTraining();
                    }
                }
            }
            // Select folder button
            else if (g_button_select_dir.contains(Point(x, y))) {
                string folder = openFolderDialog();
                if (!folder.empty()) {
                    g_training_dir = folder;
                    g_cache_valid = false; // сбрасываем кэш путей
                    bool has_images = false;
                    try {
                        for (const auto& entry : fs::recursive_directory_iterator(g_training_dir)) {
                            if (entry.is_regular_file()) {
                                string p = entry.path().string();
                                if (p.find(".jpg") != string::npos || p.find(".png") != string::npos) {
                                    has_images = true;
                                    break;
                                }
                            }
                        }
                    } catch (...) {}
                    if (!has_images) {
                        pushNotification("Selected folder has no images (.jpg, .png)");
                    }
                    ensureClassNamesFromTrainingDir();
                }
            }
            // Fast / Full mode toggle
            else if (g_button_mode.contains(Point(x, y))) {
                g_fast_mode = !g_fast_mode;
            }
        }
    }
}

string openFileDialog() {
#ifdef _WIN32
    char szFile[260] = { 0 };
    OPENFILENAMEA ofn;
    ZeroMemory(&ofn, sizeof(ofn));
    ofn.lStructSize = sizeof(ofn);
    ofn.lpstrFile = szFile;
    ofn.nMaxFile = sizeof(szFile);
    ofn.lpstrFilter = "Images\0*.jpg;*.png;*.bmp\0All\0*.*\0";
    ofn.nFilterIndex = 1;
    ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;
    if (GetOpenFileNameA(&ofn)) return string(szFile);
#endif
    return "";
}

void loadImageAndClassify(const string& path) {
    g_status_text = "Loading image...";
    drawMainScreen();
    waitKey(10);
    
    try {
        g_current_image = imread(path);
        if (g_current_image.empty()) {
            pushNotification("Error: Failed to load image");
            g_status_text = "Error: Failed to load image";
            g_classification_result = "";
            g_predicted_class_id = -1;
            g_class_metrics.clear();
            drawMainScreen();
            return;
        }
        
        // Конвертируем в BGR для корректного отображения (если изображение в градациях серого)
        if (g_current_image.channels() == 1) {
            cvtColor(g_current_image, g_current_image, COLOR_GRAY2BGR);
        }
        
        g_status_text = "Classifying...";
        drawMainScreen();
        waitKey(10);
        
        if (g_classifier) {
            ensureClassNamesFromTrainingDir();
            // Classify image using scientific methods (structure analysis, fractal, chaos theory, etc.)
            auto result = g_classifier->classifyWithStructureAnalysis(g_current_image);
            int class_id = result.first;
            g_predicted_class_id = class_id;
            g_classification_confidence = result.second;
            g_classification_result = g_classifier->getClassName(class_id);
            
            // Use metrics from last training (g_final_class_metrics) to avoid blocking UI.
            // getClassMetrics on full dataset would freeze for minutes with 2000+ images.
            g_class_metrics.clear();
            
            g_status_text = "Classification complete";
        } else {
            pushNotification("Error: Classifier not initialized");
            g_status_text = "Error: Classifier not initialized";
        }
    } catch (const exception& e) {
        pushNotification(string("Error: ") + e.what());
        g_status_text = string("Error: ") + e.what();
        g_classification_result = "";
        g_predicted_class_id = -1;
        g_class_metrics.clear();
        cerr << "loadImageAndClassify exception: " << e.what() << endl;
    } catch (...) {
        pushNotification("Error: Unknown error during classification");
        g_status_text = "Error: Unknown error during classification";
        g_classification_result = "";
        g_predicted_class_id = -1;
        g_class_metrics.clear();
        cerr << "loadImageAndClassify: unknown exception" << endl;
    }
    
    drawMainScreen();
}

void reclassifyCurrentImage() {
    if (g_current_image.empty() || !g_classifier) return;
    try {
        auto result = g_classifier->classifyWithStructureAnalysis(g_current_image);
        int class_id = result.first;
        g_predicted_class_id = class_id;
        g_classification_confidence = result.second;
        g_classification_result = g_classifier->getClassName(class_id);
    } catch (...) {}
}

void startTraining() {
    if (g_training_active) {
        return; // Already training
    }
    
    g_training_active = true;
    g_train_progress = 0.0;
    g_current_epoch = 0;
    g_current_stage = 0;
    auto& cfg = Config::getInstance();
    cfg.load("config.json");
    g_total_epochs = cfg.getInt("training_epochs", g_fast_mode ? 120 : 250);
    g_batch_size = cfg.getInt("batch_size", 8);
    g_learning_rate = cfg.getDouble("learning_rate", 0.001);
    g_training_status = "Preparing training data...";
    g_current_loss = 0.0;
    g_current_accuracy = 0.0;
    g_processed_images = 0;
    g_total_images = 0;
    g_training_start_time = chrono::steady_clock::now();
    
    thread([]() {
        try {
        // Открываем лог сразу (консоль + файл, UI не заменяется)
        char logname[64] = {0};
        {
            auto now = chrono::system_clock::now();
            time_t t = chrono::system_clock::to_time_t(now);
            tm tm_buf;
            localtime_s(&tm_buf, &t);
            strftime(logname, sizeof(logname), "training_%Y%m%d_%H%M%S.log", &tm_buf);
            lock_guard<mutex> lock(g_log_mutex);
            g_training_log.open(logname);
            if (g_training_log.is_open()) {
                g_training_log << "=== Training log " << logname << " ===" << endl;
            }
        }
        g_last_logged_epoch = -1;
        cout << "[Train] Log file: " << logname << endl;
        
        // Collect training data with proper class mapping (с кэшированием)
        vector<string> paths;
        vector<int> labels;
        
        try {
            if (g_cache_valid && g_cached_dir == g_training_dir) {
                // Используем кэш, если директория не изменилась
                paths = g_cached_paths;
                labels = g_cached_labels;
            } else {
                // Collect folder names, sort for deterministic class ID mapping
                set<string> folder_names;
                for (const auto& entry : fs::recursive_directory_iterator(g_training_dir)) {
                    if (entry.is_directory()) {
                        string fn = entry.path().filename().string();
                        if (fn != "training_images" && fn != "from_folder")
                            folder_names.insert(fn);
                    }
                }
                vector<string> sorted_folders(folder_names.begin(), folder_names.end());
                sort(sorted_folders.begin(), sorted_folders.end());
                map<string, int> folder_to_class;
                for (size_t i = 0; i < sorted_folders.size(); ++i)
                    folder_to_class[sorted_folders[i]] = static_cast<int>(i);
                
                // Second pass: collect files with class labels
                for (const auto& entry : fs::recursive_directory_iterator(g_training_dir)) {
                    if (entry.is_regular_file()) {
                        string p = entry.path().string();
                        if (p.find(".jpg") != string::npos || p.find(".png") != string::npos) {
                            paths.push_back(p);
                            // Extract class from folder name
                            fs::path folder_path = entry.path().parent_path();
                            string folder_name = folder_path.filename().string();
                            int class_id = (folder_to_class.find(folder_name) != folder_to_class.end()) 
                                         ? folder_to_class[folder_name] : 0;
                            labels.push_back(class_id);
                        }
                    }
                }
                
                // Обновляем кэш
                g_cached_paths = paths;
                g_cached_labels = labels;
                g_cached_dir = g_training_dir;
                g_cache_valid = true;
            }
        } catch (const exception& e) {
            pushNotification("Error: " + string(e.what()));
            g_training_status = "Error: " + string(e.what());
            logTraining("ERROR: " + string(e.what()));
            g_training_active = false;
            { lock_guard<mutex> lock(g_log_mutex); if (g_training_log.is_open()) g_training_log.close(); }
            g_app_state = STATE_MAIN;
            return;
        }
        
        if (paths.empty()) {
            pushNotification("Error: No training data found in " + g_training_dir);
            g_training_status = "Error: No training data found in " + g_training_dir;
            logTraining("ERROR: No training data in " + g_training_dir);
            g_training_active = false;
            { lock_guard<mutex> lock(g_log_mutex); if (g_training_log.is_open()) g_training_log.close(); }
            g_app_state = STATE_MAIN;
            return;
        }
        
        g_total_images = static_cast<int>(paths.size());
        g_training_status = "Starting training...";
        
        {
            lock_guard<mutex> lock(g_log_mutex);
            if (g_training_log.is_open()) {
                g_training_log << "Images: " << g_total_images << ", Epochs: " << g_total_epochs << endl;
            }
        }
        logTraining("Loading data: " + to_string(g_total_images) + " images, " + to_string(g_total_epochs) + " epochs");
        
        // Убедиться, что g_model_base установлен (для сохранения в правильную папку)
        if (g_model_base.empty()) {
            g_model_base = (fs::exists("x64/Release") || fs::exists("x64\\Release")) 
                ? "x64/Release/best_model" : "best_model";
        }
        
        // Load existing model if available
        string load_path = g_model_base.empty() ? "best_model" : g_model_base;
        if (fs::exists(load_path + ".member0") || fs::exists(load_path + ".classes")) {
            g_training_status = "Loading existing model...";
            logTraining("Loading existing model...");
            g_classifier->loadModel(load_path);
        }
        
        // Установить имена классов из имён папок (перезаписывает пустые после loadModel)
        map<int, string> label_to_folder;
        for (size_t i = 0; i < paths.size(); ++i) {
            int lid = labels[i];
            if (label_to_folder.find(lid) == label_to_folder.end()) {
                fs::path p(paths[i]);
                string folder_name = p.parent_path().filename().string();
                label_to_folder[lid] = folder_name;
            }
        }
        if (!label_to_folder.empty()) {
            int max_id = 0;
            for (const auto& [id, name] : label_to_folder) {
                max_id = max(max_id, id);
            }
            vector<string> ordered_names(max_id + 1);
            for (const auto& [id, name] : label_to_folder) {
                ordered_names[id] = name;
            }
            g_classifier->setClassNames(ordered_names);
        }
        
        // Start training
        g_training_status = "Preprocessing...";
        logTraining("Training started...");
        g_training_completed = false;
        g_training_history.clear();
        g_current_epoch = 0;
        g_current_stage = 0;
        g_train_progress = 0.0;
        g_problem_class_progress = 0.0;
        g_quality_control_progress = 0.0;
        g_stage4_epoch_count = 0;
        
        // Поток мониторинга прогресса: копируем history (гонка: trainOnDataset модифицирует её)
        thread([]() {
            while (g_training_active) {
                try {
                    auto history_copy = g_classifier->getTrainingHistory();
                    if (!history_copy.empty()) {
                        const TrainingStats& s = history_copy.back();
                        {
                            lock_guard<mutex> lock(g_ui_mutex);
                            if (g_training_status == "Preprocessing...") {
                                g_training_status = "Training in progress...";
                            }
                            g_current_epoch = s.epoch;
                            g_current_stage = s.stage;
                            g_current_loss = s.loss;
                            g_current_accuracy = s.accuracy;
                            g_processed_images = s.samples_processed;
                            if (g_total_epochs > 0) {
                                int total_epochs = g_total_epochs + g_stage3_epochs;
                                if (s.stage == 2) {
                                    int stage2_count = 0;
                                    for (const auto& h : history_copy) if (h.stage == 2) stage2_count++;
                                    g_train_progress = min(1.0, min(stage2_count, g_total_epochs) / static_cast<double>(total_epochs));
                                } else if (s.stage >= 3) {
                                    g_train_progress = min(1.0, (g_total_epochs + s.epoch) / static_cast<double>(total_epochs));
                                }
                            }
                            if (s.stage == 4) {
                                int stage4_count = 0;
                                for (const auto& h : history_copy) {
                                    if (h.stage == 4) stage4_count++;
                                }
                                g_stage4_epoch_count = stage4_count;
                                g_problem_class_progress = min(1.0, stage4_count / static_cast<double>(g_stage4_total_epochs));
                            }
                            if (s.stage == 1 && s.epoch > 0) {
                                g_quality_control_progress = min(1.0, static_cast<double>(s.samples_processed) / s.epoch);
                            } else if (s.stage >= 2) {
                                g_quality_control_progress = 1.0;
                            }
                        }
                        // Вывод в консоль и лог при смене эпохи (UI не трогаем)
                        if (s.epoch > g_last_logged_epoch) {
                            g_last_logged_epoch = s.epoch;
                            stringstream ss;
                            ss << "Epoch " << s.epoch << "/" << g_total_epochs 
                               << " Loss=" << fixed << setprecision(4) << s.loss 
                               << " Acc=" << setprecision(2) << (s.accuracy * 100) << "%";
                            logTraining(ss.str());
                        }
                    }
                } catch (...) {
                    // Игнорируем ошибки во время обучения
                }
                this_thread::sleep_for(chrono::milliseconds(200));
            }
        }).detach();
        
        // Fast mode: отключаем анализ структуры (SVM) — ускоряет обучение
        bool was_structure = g_classifier->isStructureAnalysisEnabled();
        if (g_fast_mode) {
            g_classifier->enableStructureAnalysis(false);
            logTraining("Fast mode: structure analysis disabled");
        }

        // Применяем learning rate из config; для малых датасетов (<600) — ещё ниже для стабильности
        double lr = g_learning_rate;
        if (paths.size() < 600) {
            lr = min(lr, 0.0003);
            logTraining("Small dataset (" + to_string(paths.size()) + " images): using reduced LR=" + to_string(lr));
        }
        g_classifier->setLearningRate(lr);

        // Параметры: focal_loss, oversampling, extended_aug, stage4, label_smoothing, dropout, adaptive_clip, mixup, cosine_annealing
        // Включены: label_smoothing, focal_loss для баланса классов; mixup для регуляризации
        g_classifier->trainOnDataset(paths, labels, g_total_epochs,
            true, true, true, true, true, true, true, true, true);
        
        if (g_fast_mode) g_classifier->enableStructureAnalysis(was_structure);
        
        g_training_status = "Saving model...";
        g_classifier->saveModel(g_model_base.empty() ? "best_model" : g_model_base);
        saveTrainingDir();
        
        // Получаем данные обучения БЕЗ lock (getClassMetrics может занимать минуты)
        vector<TrainingStats> history = g_classifier->getTrainingHistory();
        vector<int> problem_classes = g_classifier->getProblemClasses();
        vector<NeuralNetwork::ClassMetrics> final_metrics;
        try {
            final_metrics = g_classifier->getClassMetrics(paths, labels);
        } catch (...) {
            cerr << "[Training] getClassMetrics failed, saving without per-class metrics" << endl;
        }
        
        // Вычисляем финальные метрики из истории
        double final_acc = 0.0;
        double final_macro = 0.0;
        int display_num = g_classifier ? static_cast<int>(g_classifier->getClassList().size()) : 0;
        if (!history.empty()) {
            final_acc = history.back().accuracy;
            if (!final_metrics.empty()) {
                double total_f1 = 0.0;
                int classes_with_data = 0;
                for (const auto& metric : final_metrics) {
                    bool has_data = (metric.true_positives + metric.false_negatives > 0) || 
                                   (metric.true_positives + metric.false_positives > 0);
                    if (has_data) {
                        total_f1 += metric.f1_score;
                        classes_with_data++;
                    }
                }
                final_macro = (classes_with_data > 0) ? (total_f1 / classes_with_data) : final_acc;
            } else {
                final_macro = final_acc;
            }
        }
        if (display_num == 0 && !final_metrics.empty()) {
            int max_id = 0;
            for (const auto& m : final_metrics) max_id = max(max_id, m.class_id);
            display_num = max_id + 1;
        }
        
        // Кратко берём lock — всегда сохраняем (даже при пустых метриках)
        {
            lock_guard<mutex> lock(g_ui_mutex);
            g_training_history = move(history);
            g_problem_classes = move(problem_classes);
            g_final_class_metrics = move(final_metrics);
            g_final_accuracy = final_acc;
            g_final_macro_f1 = final_macro;
            g_final_weighted_f1 = final_acc;
            g_training_completed = true;
            g_display_accuracy = final_acc;
            g_display_f1 = final_macro;
            g_display_num_classes = display_num;
            saveBestModelMetrics();
            saveTrainingDir();
            saveTrainingHistory();
        }
        
        g_training_status = "Training finished!";
        g_train_progress = 1.0;
        g_quality_control_progress = 1.0;
        if (!g_problem_classes.empty()) {
            g_problem_class_progress = 1.0;
            g_stage4_epoch_count = g_stage4_total_epochs;
        }
        
        // Финальная запись в консоль и лог
        {
            stringstream ss;
            ss << "Finished! Accuracy=" << fixed << setprecision(2) << (g_final_accuracy * 100) << "%";
            logTraining(ss.str());
        }
        {
            lock_guard<mutex> lock(g_log_mutex);
            if (g_training_log.is_open()) {
                g_training_log << "=== Training complete ===" << endl;
                g_training_log.close();
            }
        }
        
        g_training_active = false;
        
        // Return to main screen immediately (program stays open)
        g_app_state = STATE_MAIN;
        this_thread::sleep_for(std::chrono::milliseconds(100));
        } catch (const exception& e) {
            cerr << "[Training] Fatal error: " << e.what() << endl;
            pushNotification("Error: " + string(e.what()));
            g_training_status = "Error: " + string(e.what());
            g_training_active = false;
            { lock_guard<mutex> lock(g_log_mutex); if (g_training_log.is_open()) g_training_log.close(); }
            saveTrainingDir();
            g_app_state = STATE_MAIN;
        } catch (...) {
            cerr << "[Training] Unknown fatal error" << endl;
            pushNotification("Error: Unknown error");
            g_training_status = "Error: Unknown error";
            g_training_active = false;
            { lock_guard<mutex> lock(g_log_mutex); if (g_training_log.is_open()) g_training_log.close(); }
            saveTrainingDir();
            g_app_state = STATE_MAIN;
        }
    }).detach();
}

int main(int argc, char* argv[]) {
#ifdef _WIN32
    SetUnhandledExceptionFilter(crashHandler);
    HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
    if (hConsole != INVALID_HANDLE_VALUE) {
        SetConsoleTextAttribute(hConsole, FOREGROUND_GREEN | FOREGROUND_INTENSITY);
    }
#endif
    cout << "==========================================" << endl;
    cout << "   Image Classifier System                " << endl;
    cout << "==========================================" << endl;
    
    // Инициализация CUDA для обучения на видеокарте
    CudaAccelerator::initialize();
    cout << "Ускорение: " << (CudaAccelerator::isAvailable() ? "CUDA (GPU)" : "CPU") << endl;
    
    // Try to load existing model and read num_classes from metadata
    int num_classes = 10;
    string model_paths[] = {"best_model", "x64/Release/best_model", "x64/Debug/best_model"};
    for (const auto& base : model_paths) {
        if (fs::exists(base + ".member0")) {
            int in_size, out_size;
            double lr;
            vector<int> hidden;
            if (NeuralNetwork::loadModelMetadata(base + ".member0", in_size, hidden, out_size, lr)) {
                num_classes = out_size;
                break;
            }
        }
    }
    
    g_classifier = new UniversalImageClassifier(num_classes, 32);
    
    bool model_loaded = false;
    for (const auto& base : model_paths) {
        if (fs::exists(base + ".member0") || fs::exists(base + ".classes")) {
            cout << "Loading existing model from " << base << "..." << endl;
            try {
                g_classifier->loadModel(base);
                g_status_text = "Model loaded successfully";
                model_loaded = true;
                g_model_base = base;
                g_classifier->enableStructureAnalysis(false);
                break;
            } catch (...) {
                cout << "Failed to load model from " << base << ", trying next..." << endl;
            }
        }
    }
    if (!model_loaded) {
        g_status_text = "No model found. Please train first.";
        g_classifier->enableStructureAnalysis(true);
    }
    
    loadTrainingDir(argc > 0 ? argv[0] : nullptr);
    // CLI: --train "path" — задать папку и автостарт обучения
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--train") == 0 && i + 1 < argc) {
            g_training_dir = fs::absolute(argv[i + 1]).string();
            g_cache_valid = false;
            g_auto_start_training = true;
            saveTrainingDir();
            cout << "[CLI] Auto-train folder: " << g_training_dir << endl;
            break;
        }
    }
    loadBestModelMetrics();
    ensureClassNamesFromTrainingDir();
    loadTrainingHistory();
    if (g_display_num_classes == 0 && g_classifier) {
        g_display_num_classes = static_cast<int>(g_classifier->getClassList().size());
    }
    
    // Create window
    namedWindow("Image Classifier", WINDOW_NORMAL);
    resizeWindow("Image Classifier", WINDOW_WIDTH, WINDOW_HEIGHT);
    setMouseCallback("Image Classifier", onMouse);
#ifdef _WIN32
    disableWindowClose("Image Classifier");
#endif
    
    // Main loop (X и WM_CLOSE перехвачены — окно не закрывается)
    while (true) {
        try {
            double vis = -1;
            try { vis = getWindowProperty("Image Classifier", WND_PROP_VISIBLE); } catch (...) {}
            if (vis < 0) {
                namedWindow("Image Classifier", WINDOW_NORMAL);
                resizeWindow("Image Classifier", WINDOW_WIDTH, WINDOW_HEIGHT);
                setMouseCallback("Image Classifier", onMouse);
#ifdef _WIN32
                g_closeDisabled = false;  // пересоздали окно — снова привязываем
                disableWindowClose("Image Classifier");
#endif
            }
            if (g_app_state == STATE_MAIN) {
                drawMainScreen();
            } else {
                drawTrainingScreen();
            }
            if (g_auto_start_training && !g_training_active) {
                g_auto_start_training = false;
                g_app_state = STATE_TRAINING;
                g_training_status = "Ready to train";
                g_training_completed = false;
                startTraining();
            }
            waitKey(1);
        } catch (const exception& e) {
            cerr << "[Main loop] Exception: " << e.what() << endl;
        } catch (...) {
            cerr << "[Main loop] Unknown exception" << endl;
        }
    }
    
    destroyAllWindows();
    delete g_classifier;
    return 0;
}
