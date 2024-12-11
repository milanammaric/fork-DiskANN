#include "node_cache.h"
#include "utils.h"

namespace diskann
{
template <typename T, typename LabelT> NodeCache<T, LabelT>::NodeCache(u_int32_t max_degree, u_int32_t aligned_dim): _max_degree(max_degree), _aligned_dim(aligned_dim)
{};

template <typename T, typename LabelT> NodeCache<T, LabelT>::~NodeCache()
{
    if (_nhood_cache_dram_buf != nullptr)
    {
        delete[] _nhood_cache_dram_buf;
        diskann::aligned_free(_coord_cache_dram_buf);
    }

    // TODO: Change this so that buffer is deallocated in cxl
    if (_nhood_cache_cxl_buf != nullptr)
    {
        delete[] _nhood_cache_cxl_buf;
        diskann::aligned_free(_coord_cache_cxl_buf);
    }
};

 template <typename T, typename LabelT> void NodeCache<T, LabelT>::load_cache_list(std::vector<uint32_t> &node_list, double dram_percentage, 
    std::function<std::vector<bool> (const std::vector<uint32_t> &, std::vector<T *> &, std::vector<std::pair<uint32_t, uint32_t *>> &)> read_nodes)
{
    if(_nhood_cache_dram_buf != nullptr || _nhood_cache_cxl_buf != nullptr)
    {
        std::stringstream stream;
        stream << "Cache already loaded. Cannot load cache again.";
        print_error_and_terminate(stream);
    }

    diskann::cout << "Loading the cache list into memory.." << std::flush;
    _num_cached_nodes_dram = (size_t)(dram_percentage * node_list.size());
    _num_cached_nodes_cxl = node_list.size() - _num_cached_nodes_dram;

    if (_num_cached_nodes_dram > 0)
    {
         // Allocate space for neighborhood cache in dram
        _nhood_cache_dram_buf = new uint32_t[_num_cached_nodes_dram * (_max_degree + 1)];
        memset(_nhood_cache_dram_buf, 0, _num_cached_nodes_dram * (_max_degree + 1));

        // Allocate space for coordinate cache in dram
        size_t coord_cache_buf_len = _num_cached_nodes_dram * _aligned_dim;
        diskann::alloc_aligned((void **)&_coord_cache_dram_buf, coord_cache_buf_len * sizeof(T), 8 * sizeof(T));
        memset(_coord_cache_dram_buf, 0, coord_cache_buf_len * sizeof(T));
    }

    // TODO: Change this so that buffer is allocated in cxl
    if (_num_cached_nodes_cxl > 0)
    {
        // Allocate space for neighborhood cache in cxl
        _nhood_cache_cxl_buf = new uint32_t[_num_cached_nodes_cxl * (_max_degree + 1)];
        memset(_nhood_cache_cxl_buf, 0, _num_cached_nodes_cxl * (_max_degree + 1));

        // Allocate space for coordinate cache in cxl
        size_t coord_cache_buf_len = _num_cached_nodes_cxl * _aligned_dim;
        diskann::alloc_aligned((void **)&_coord_cache_cxl_buf, coord_cache_buf_len * sizeof(T), 8 * sizeof(T));
        memset(_coord_cache_cxl_buf, 0, coord_cache_buf_len * sizeof(T));
    }

    size_t BLOCK_SIZE = 8;
    size_t num_blocks = DIV_ROUND_UP(node_list.size(), BLOCK_SIZE);
    for (size_t block = 0; block < num_blocks; block++)
    {
        size_t start_idx = block * BLOCK_SIZE;
        size_t end_idx = (std::min)(node_list.size(), (block + 1) * BLOCK_SIZE);

        // Copy offset into buffers to read into
        std::vector<uint32_t> nodes_to_read;
        std::vector<T *> coord_buffers;
        std::vector<std::pair<uint32_t, uint32_t *>> nbr_buffers;
        for (size_t node_idx = start_idx; node_idx < end_idx; node_idx++)
        {
            if (node_idx < _num_cached_nodes_dram)
            {
                nodes_to_read.push_back(node_list[node_idx]);
                coord_buffers.push_back(_coord_cache_dram_buf + node_idx * _aligned_dim);
                _nodes_in_dram.insert(node_idx);
                nbr_buffers.emplace_back(0, _nhood_cache_dram_buf + node_idx * (_max_degree + 1));
            }
            else
            {
                nodes_to_read.push_back(node_list[node_idx]);
                coord_buffers.push_back(_coord_cache_cxl_buf + (node_idx - _num_cached_nodes_dram) * _aligned_dim);
                _nodes_in_cxl.insert(node_idx);
                nbr_buffers.emplace_back(0, _nhood_cache_cxl_buf + (node_idx - _num_cached_nodes_dram) * (_max_degree + 1));
            }

            // issue the reads
            auto read_status = read_nodes(nodes_to_read, coord_buffers, nbr_buffers);

            // check for success and insert into the cache.
            for (size_t i = 0; i < read_status.size(); i++)
            {
                if (read_status[i] == true)
                {
                    _coord_cache.insert(std::make_pair(nodes_to_read[i], coord_buffers[i]));
                    _nhood_cache.insert(std::make_pair(nodes_to_read[i], nbr_buffers[i]));
                }
            }
        }
    }
    diskann::cout << "..done." << std::endl;
};

template <typename T, typename LabelT> void NodeCache<T, LabelT>::reload_cache_list(std::unordered_set<uint32_t> &nodes_to_store_in_dram, std::unordered_set<uint32_t> &nodes_to_store_in_cxl, 
    std::function<std::vector<bool> (const std::vector<uint32_t> &, std::vector<T *> &, std::vector<std::pair<uint32_t, uint32_t *>> &)> read_nodes) {
    // TODO: Implement this function
};

template <typename T, typename LabelT> std::optional<std::pair<uint32_t, uint32_t *> > NodeCache<T, LabelT>::find_nhood(uint32_t node_id)
{
    auto it = _nhood_cache.find(node_id);
    if (it != _nhood_cache.end())
    {
        return it->second;
    }
    return std::nullopt;
};

template <typename T, typename LabelT> T* NodeCache<T, LabelT>::find_coords(uint32_t node_id)
{
    auto it = _coord_cache.find(node_id);
    if (it != _coord_cache.end())
    {
        return it->second;
    }
    return nullptr;
};

// instantiations
template class NodeCache<uint8_t>;
template class NodeCache<int8_t>;
template class NodeCache<float>;
template class NodeCache<uint8_t, uint16_t>;
template class NodeCache<int8_t, uint16_t>;
template class NodeCache<float, uint16_t>;

}// namespace diskann
