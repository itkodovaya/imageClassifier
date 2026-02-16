#include "HyperRelationalFuzzyGraph.h"
#include "ShapeAnalyzer.h"
#include <cmath>
#include <algorithm>

HyperRelationalFuzzyGraph::HyperRelationalFuzzyGraph() {
}

int HyperRelationalFuzzyGraph::addLandmark(const TopologicalLandmark& landmark) {
    landmarks.push_back(landmark);
    return static_cast<int>(landmarks.size() - 1);
}

void HyperRelationalFuzzyGraph::addRelation(const FuzzySpatialRelation& relation) {
    relations.push_back(relation);
}

void HyperRelationalFuzzyGraph::buildFromStructures(
    const std::vector<StructureRegion>& structures,
    const ShapeDescription& shape_desc) {
    
    landmarks.clear();
    relations.clear();
    
    // Создаем ориентиры из центров структур
    for (size_t i = 0; i < structures.size(); ++i) {
        TopologicalLandmark landmark;
        landmark.position = structures[i].center;
        landmark.type = structures[i].structure_type;
        landmark.confidence = structures[i].confidence;
        landmark.structure_id = static_cast<int>(i);
        landmark.features = VectorXd::Zero(5);
        landmark.features(0) = structures[i].area;
        landmark.features(1) = structures[i].prominence;
        landmark.features(2) = structures[i].center.x;
        landmark.features(3) = structures[i].center.y;
        landmark.features(4) = structures[i].confidence;
        
        addLandmark(landmark);
    }
    
    // Вычисляем пространственные отношения
    computeSpatialRelations();
}

void HyperRelationalFuzzyGraph::computeSpatialRelations() {
    relations.clear();
    
    if (landmarks.size() < 2) return;
    
    // Вычисляем максимальное расстояние для нормализации
    double max_distance = 0.0;
    for (size_t i = 0; i < landmarks.size(); ++i) {
        for (size_t j = i + 1; j < landmarks.size(); ++j) {
            float dx = landmarks[i].position.x - landmarks[j].position.x;
            float dy = landmarks[i].position.y - landmarks[j].position.y;
            double dist = sqrt(dx * dx + dy * dy);
            max_distance = max(max_distance, dist);
        }
    }
    
    if (max_distance < 1e-6) max_distance = 1.0;
    
    // Создаем отношения между всеми парами узлов
    for (size_t i = 0; i < landmarks.size(); ++i) {
        for (size_t j = i + 1; j < landmarks.size(); ++j) {
            float dx = landmarks[j].position.x - landmarks[i].position.x;
            float dy = landmarks[j].position.y - landmarks[i].position.y;
            double dist = sqrt(dx * dx + dy * dy);
            double angle = atan2(dy, dx);
            
            // Отношение "слева"
            FuzzySpatialRelation left_rel;
            left_rel.from_node = static_cast<int>(i);
            left_rel.to_node = static_cast<int>(j);
            left_rel.relation_type = "left";
            left_rel.membership = computeLeftMembership(landmarks[i], landmarks[j]);
            left_rel.distance = dist;
            left_rel.angle = angle;
            if (left_rel.membership > 0.1) {
                addRelation(left_rel);
            }
            
            // Отношение "сверху"
            FuzzySpatialRelation above_rel;
            above_rel.from_node = static_cast<int>(i);
            above_rel.to_node = static_cast<int>(j);
            above_rel.relation_type = "above";
            above_rel.membership = computeAboveMembership(landmarks[i], landmarks[j]);
            above_rel.distance = dist;
            above_rel.angle = angle;
            if (above_rel.membership > 0.1) {
                addRelation(above_rel);
            }
            
            // Отношение "близко"
            FuzzySpatialRelation near_rel;
            near_rel.from_node = static_cast<int>(i);
            near_rel.to_node = static_cast<int>(j);
            near_rel.relation_type = "near";
            near_rel.membership = computeNearMembership(dist, max_distance);
            near_rel.distance = dist;
            near_rel.angle = angle;
            if (near_rel.membership > 0.1) {
                addRelation(near_rel);
            }
            
            // Отношение "далеко"
            FuzzySpatialRelation far_rel;
            far_rel.from_node = static_cast<int>(i);
            far_rel.to_node = static_cast<int>(j);
            far_rel.relation_type = "far";
            far_rel.membership = computeFarMembership(dist, max_distance);
            far_rel.distance = dist;
            far_rel.angle = angle;
            if (far_rel.membership > 0.1) {
                addRelation(far_rel);
            }
        }
    }
}

double HyperRelationalFuzzyGraph::computeLeftMembership(
    const TopologicalLandmark& from, const TopologicalLandmark& to) const {
    
    float dx = to.position.x - from.position.x;
    // Если to находится слева от from (dx < 0), то from находится справа от to
    // Используем нечеткую функцию принадлежности
    if (dx < 0) {
        return exp(-dx * dx / (2.0 * 100.0 * 100.0)); // Гауссова функция
    }
    return 0.0;
}

double HyperRelationalFuzzyGraph::computeAboveMembership(
    const TopologicalLandmark& from, const TopologicalLandmark& to) const {
    
    float dy = to.position.y - from.position.y;
    // Если to находится выше from (dy < 0), то from находится ниже to
    if (dy < 0) {
        return exp(-dy * dy / (2.0 * 100.0 * 100.0));
    }
    return 0.0;
}

double HyperRelationalFuzzyGraph::computeConnectedMembership(
    const TopologicalLandmark& from, const TopologicalLandmark& to,
    const vector<StructureRegion>& structures) const {
    
    // Упрощенная версия: считаем соединенными, если расстояние мало
    float dx = to.position.x - from.position.x;
    float dy = to.position.y - from.position.y;
    double dist = sqrt(dx * dx + dy * dy);
    
    // Если расстояние меньше порога, считаем соединенными
    double threshold = 50.0;
    if (dist < threshold) {
        return 1.0 - (dist / threshold);
    }
    return 0.0;
}

double HyperRelationalFuzzyGraph::computeNearMembership(
    double distance, double max_distance) const {
    
    double normalized_dist = distance / max_distance;
    // Нечеткая функция "близко": чем меньше расстояние, тем больше принадлежность
    return exp(-normalized_dist * normalized_dist / (2.0 * 0.3 * 0.3));
}

double HyperRelationalFuzzyGraph::computeFarMembership(
    double distance, double max_distance) const {
    
    double normalized_dist = distance / max_distance;
    // Нечеткая функция "далеко": чем больше расстояние, тем больше принадлежность
    return 1.0 - exp(-normalized_dist * normalized_dist / (2.0 * 0.3 * 0.3));
}

VectorXd HyperRelationalFuzzyGraph::computeGraphFeatures() const {
    // Создаем вектор признаков из графа
    int feature_size = 50; // Размерность признаков
    VectorXd features = VectorXd::Zero(feature_size);
    
    int idx = 0;
    
    // Признаки узлов
    for (const auto& landmark : landmarks) {
        if (idx < feature_size - 10) {
            features(idx++) = landmark.position.x;
            features(idx++) = landmark.position.y;
            features(idx++) = landmark.confidence;
            if (landmark.features.size() > 0) {
                features(idx++) = landmark.features(0);
            }
        }
    }
    
    // Признаки ребер (отношений)
    for (const auto& relation : relations) {
        if (idx < feature_size - 5) {
            features(idx++) = relation.membership;
            features(idx++) = relation.distance;
            features(idx++) = relation.angle;
        }
    }
    
    // Статистика графа
    features(feature_size - 5) = static_cast<double>(landmarks.size());
    features(feature_size - 4) = static_cast<double>(relations.size());
    
    // Средняя степень принадлежности
    double avg_membership = 0.0;
    if (!relations.empty()) {
        for (const auto& rel : relations) {
            avg_membership += rel.membership;
        }
        avg_membership /= relations.size();
    }
    features(feature_size - 3) = avg_membership;
    
    return features;
}

double HyperRelationalFuzzyGraph::compareGraphs(
    const HyperRelationalFuzzyGraph& other) const {
    
    // Вычисляем нечеткое сходство между графами
    // Используем сравнение структуры и отношений
    
    if (landmarks.empty() && other.landmarks.empty()) return 1.0;
    if (landmarks.empty() || other.landmarks.empty()) return 0.0;
    
    // Сравниваем количество узлов и ребер
    double node_similarity = 1.0 - abs(static_cast<double>(landmarks.size() - other.landmarks.size())) 
                            / max(landmarks.size(), other.landmarks.size());
    double edge_similarity = 1.0 - abs(static_cast<double>(relations.size() - other.relations.size()))
                            / max(relations.size(), other.relations.size());
    
    // Сравниваем признаки графов
    VectorXd features1 = computeGraphFeatures();
    VectorXd features2 = other.computeGraphFeatures();
    
    int min_size = min(features1.size(), features2.size());
    double feature_similarity = 0.0;
    for (int i = 0; i < min_size; ++i) {
        double diff = abs(features1(i) - features2(i));
        feature_similarity += exp(-diff * diff);
    }
    feature_similarity /= min_size;
    
    // Комбинируем сходства
    return (node_similarity + edge_similarity + feature_similarity) / 3.0;
}

