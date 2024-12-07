#pragma once
#include <cstdint>
#include <optional>

#include "tsl/robin_map.h"


namespace diskann
{

template <typename T, typename LabelT = uint32_t> class NodeCache
{
private:
    // nhood_cache; the uint32_t in nhood_Cache are offsets into nhood_cache_buf
    unsigned *_nhood_cache_dram_buf = nullptr;
    unsigned *_nhood_cache_cxl_buf = nullptr;
    tsl::robin_map<uint32_t, std::pair<uint32_t, uint32_t *>> _nhood_cache;

    // coord_cache; The T* in coord_cache are offsets into coord_cache_buf
    T *_coord_cache_dram_buf = nullptr;
    T *_coord_cache_cxl_buf = nullptr;
    tsl::robin_map<uint32_t, T *> _coord_cache;
    
    u_int32_t _max_degree;
    u_int32_t _aligned_dim;
public: 
    NodeCache(u_int32_t max_degree, u_int32_t aligned_dim);
    ~NodeCache();

    void load_cache_list(std::vector<uint32_t> &node_list, double dram_percentage,
                         std::function<std::vector<bool>(const std::vector<uint32_t> &, std::vector<T *> &,
                                                         std::vector<std::pair<uint32_t, uint32_t *>> &)>
                             read_nodes);

    std::optional<std::pair<uint32_t, uint32_t *> > find_nhood(uint32_t node_id);
    T* find_coords(uint32_t node_id);
};

}; // namespace diskann

