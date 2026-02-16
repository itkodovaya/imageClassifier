#ifndef CONFIG_H
#define CONFIG_H

#include <string>
#include <map>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>

// Simple configuration loader (JSON-like and key=value formats)
// Plan item 5.3: Configuration and settings
class Config {
public:
    static Config& getInstance() {
        static Config instance;
        return instance;
    }

    bool load(const std::string& path = "config.json") {
        std::ifstream f(path);
        if (!f.is_open()) return false;
        std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        f.close();
        return parse(content);
    }

    bool save(const std::string& path = "config.json") {
        std::ofstream f(path);
        if (!f.is_open()) return false;
        f << "{\n";
        bool first = true;
        for (const auto& [k, v] : values_) {
            if (!first) f << ",\n";
            f << "  \"" << escapeJson(k) << "\": \"" << escapeJson(v) << "\"";
            first = false;
        }
        f << "\n}\n";
        f.close();
        return true;
    }

    std::string get(const std::string& key, const std::string& default_val = "") const {
        auto it = values_.find(key);
        return (it != values_.end()) ? it->second : default_val;
    }

    int getInt(const std::string& key, int default_val = 0) const {
        std::string s = get(key);
        if (s.empty()) return default_val;
        try { return std::stoi(s); } catch (...) { return default_val; }
    }

    double getDouble(const std::string& key, double default_val = 0) const {
        std::string s = get(key);
        if (s.empty()) return default_val;
        try { return std::stod(s); } catch (...) { return default_val; }
    }

    bool getBool(const std::string& key, bool default_val = false) const {
        std::string s = get(key);
        std::transform(s.begin(), s.end(), s.begin(), ::tolower);
        if (s == "true" || s == "1") return true;
        if (s == "false" || s == "0") return false;
        return default_val;
    }

    void set(const std::string& key, const std::string& value) {
        values_[key] = value;
    }

    void set(const std::string& key, int value) {
        values_[key] = std::to_string(value);
    }

    void set(const std::string& key, double value) {
        values_[key] = std::to_string(value);
    }

    void set(const std::string& key, bool value) {
        values_[key] = value ? "true" : "false";
    }

private:
    std::map<std::string, std::string> values_;

    Config() = default;
    ~Config() = default;
    Config(const Config&) = delete;
    Config& operator=(const Config&) = delete;

    bool parse(const std::string& content) {
        size_t pos = 0;
        while ((pos = content.find('"', pos)) != std::string::npos) {
            size_t key_end = content.find('"', pos + 1);
            if (key_end == std::string::npos) break;
            std::string key = content.substr(pos + 1, key_end - pos - 1);
            size_t colon_pos = content.find(':', key_end);
            if (colon_pos == std::string::npos) break;
            pos = colon_pos + 1;
            while (pos < content.size() && (content[pos] == ' ' || content[pos] == '\t')) pos++;
            if (pos >= content.size()) break;
            std::string val;
            if (content[pos] == '"') {
                size_t val_end = content.find('"', pos + 1);
                if (val_end == std::string::npos) break;
                val = content.substr(pos + 1, val_end - pos - 1);
                pos = val_end + 1;
            } else {
                size_t val_start = pos;
                while (pos < content.size() && content[pos] != ',' && content[pos] != '}' && content[pos] != '\n') pos++;
                val = content.substr(val_start, pos - val_start);
                while (!val.empty() && (val.back() == ' ' || val.back() == '\t')) val.pop_back();
            }
            values_[key] = val;
        }
        return true;
    }

    static std::string escapeJson(const std::string& s) {
        std::string r;
        for (char c : s) {
            if (c == '"') r += "\\\"";
            else if (c == '\\') r += "\\\\";
            else r += c;
        }
        return r;
    }
};

#endif
