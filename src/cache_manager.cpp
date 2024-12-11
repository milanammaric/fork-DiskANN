#include "cache_manager.h"
#include "utils.h"
#include <queue>

namespace diskann
{
    template <typename T, typename LabelT>
    CacheManager<T, LabelT>::CacheManager(u_int32_t max_degree, u_int32_t aligned_dim, double dram_percentage)
        : _dram_percentage(dram_percentage), _node_cache(max_degree, aligned_dim)
    {
    }
    
    template <typename T, typename LabelT>
    void CacheManager<T, LabelT>::get_nodes_to_cache(std::unordered_set<uint32_t> &nodes_to_cache_in_dram,
                                                    std::unordered_set<uint32_t> &nodes_to_cache_in_cxl)
    {
        using NodeFrequencyPair = std::pair<int, int>; // {frequency, nodeId}
        auto compare = [](const NodeFrequencyPair& a, const NodeFrequencyPair& b) {
            return a.first > b.first; // Min-heap based on frequency
        };
        std::priority_queue<NodeFrequencyPair, std::vector<NodeFrequencyPair>, decltype(compare)> minHeap(compare);

        // Traverse the map and maintain a heap of size n
        for (const auto& entry : _node_access_frequency) {
            int nodeId = entry.first;
            int frequency = entry.second;

            minHeap.emplace(frequency, nodeId);

            if (minHeap.size() > _cache_size) {
                minHeap.pop();
            }
        }

        size_t num_cached_nodes_dram = (size_t)(_dram_percentage * _cache_size);
        while (!minHeap.empty()) {
            if (nodes_to_cache_in_dram.size() < num_cached_nodes_dram) {
                nodes_to_cache_in_dram.insert(minHeap.top().second);
            } else {
                nodes_to_cache_in_cxl.insert(minHeap.top().second);
            }
            minHeap.pop();
        }
    }

    template <typename T, typename LabelT>
    void CacheManager<T, LabelT>::load_cache_list(std::vector<uint32_t> &node_list,
                                                  std::function<std::vector<bool>(const std::vector<uint32_t> &, std::vector<T *> &,
                                                                                  std::vector<std::pair<uint32_t, uint32_t *>> &)>
                                                      read_nodes)
    {
        if (_cache_loaded)
        {
            std::stringstream stream;
            stream << "Cache already loaded. Cannot load cache again.";
            print_error_and_terminate(stream);
        }

        _cache_size = node_list.size();
        _read_nodes = read_nodes;
        _node_cache.load_cache_list(node_list, _dram_percentage, _read_nodes);
        _cache_loaded = true;
    }

    template <typename T, typename LabelT>
    std::optional<std::pair<uint32_t, uint32_t *>> CacheManager<T, LabelT>::find_nhood(uint32_t node_id)
    {
        // Update the access frequency of the node
        auto it = _node_access_frequency.find(node_id);
        if (it == _node_access_frequency.end())
        {
            _node_access_frequency[node_id] = 0;
        }
        _node_access_frequency[node_id] = _node_access_frequency[node_id] + 1;

        // Check if the node is in the cache
        return _node_cache.find_nhood(node_id);
    }

    template <typename T, typename LabelT>
    T *CacheManager<T, LabelT>::find_coords(uint32_t node_id)
    {
        return _node_cache.find_coords(node_id);
    }

    template <typename T, typename LabelT>
    void CacheManager<T, LabelT>::record_new_query()
    {
        _epoch_counter++;
        if (_epoch_counter == QUERIES_IN_EPOCH)
        {
            // Reload cache
            std::unordered_set<uint32_t> nodes_to_store_in_dram, nodes_to_store_in_cxl;
            get_nodes_to_cache(nodes_to_store_in_dram, nodes_to_store_in_cxl);
            _node_cache.reload_cache_list(nodes_to_store_in_dram, nodes_to_store_in_cxl, _read_nodes);

            // Reset the access frequency of all nodes
            _epoch_counter = 0;
        }
    }

    template class CacheManager<uint8_t>;
    template class CacheManager<int8_t>;
    template class CacheManager<float>;
    template class CacheManager<uint8_t, uint16_t>;
    template class CacheManager<int8_t, uint16_t>;
    template class CacheManager<float, uint16_t>;
}; // namespace diskann
