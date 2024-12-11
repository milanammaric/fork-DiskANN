#pragma once

#include "node_cache.h"

#include "tsl/robin_map.h"

#define QUERIES_IN_EPOCH 1000

namespace diskann
{

template <typename T, typename LabelT = uint32_t> class CacheManager
{
private:
    tsl::robin_map<uint32_t, uint32_t> _node_access_frequency;
    uint32_t _epoch_counter = 0;

    double _dram_percentage;
    uint32_t _cache_size;
    bool _cache_loaded;
    std::function<std::vector<bool>(const std::vector<uint32_t> &, std::vector<T *> &,
                                    std::vector<std::pair<uint32_t, uint32_t *>> &)>
        _read_nodes;
    NodeCache<T, LabelT> _node_cache;
    
    void get_nodes_to_cache(std::unordered_set<uint32_t> &nodes_to_cache_in_dram,
                            std::unordered_set<uint32_t> &nodes_to_cache_in_cxl);

public:
    CacheManager(u_int32_t max_degree, u_int32_t aligned_dim, double dram_percentage = 0.5);


    void load_cache_list(std::vector<uint32_t> &node_list, std::function<std::vector<bool>(const std::vector<uint32_t> &, std::vector<T *> &,
                                                    std::vector<std::pair<uint32_t, uint32_t *>> &)>
                        read_nodes);

    std::optional<std::pair<uint32_t, uint32_t *> > find_nhood(uint32_t node_id);
    T* find_coords(uint32_t node_id);

    void record_new_query();
};

}; // namespace diskann
