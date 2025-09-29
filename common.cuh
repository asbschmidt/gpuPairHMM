#ifndef COMMON_CUH
#define COMMON_CUH

#include <algorithm>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <string>
#include <optional>
#include <cassert>
#include <iostream>

#include <thrust/device_malloc_allocator.h>

struct Options{
    bool checkResults = false;
    bool verbose = false;
    int transferBatchsize = 1000000;
    int fileBatchsize = 10000000;
    int queue_depth = 3;
    std::string inputfile = "";
    std::string outputfile = "";
};

template <class T>
struct PinnedAllocator {
    using value_type = T;

    // PinnedAllocator() = default;

    // template <class U>
    // constexpr Mallocator(const Mallocator<U>&) noexcept {}

    T* allocate(size_t elements){
        T* ptr{};
        cudaError_t err = cudaMallocHost(&ptr, elements * sizeof(T));
        if(err != cudaSuccess){
            std::cerr << "SimpleAllocator: Failed to allocate " << (elements) << " * " << sizeof(T)
                        << " = " << (elements * sizeof(T))
                        << " bytes using cudaMallocHost!\n";

            throw std::bad_alloc();
        }

        assert(ptr != nullptr);

        return ptr;
    }

    void deallocate(T* ptr, size_t /*n*/){
        cudaFreeHost(ptr);
    }
};

template<class T>
struct ThrustCudaMallocAsyncAllocator : thrust::device_malloc_allocator<T> {
    using value_type = T;
    using super_t = thrust::device_malloc_allocator<T>;

    using pointer = typename super_t::pointer;
    using size_type = typename super_t::size_type;
    using reference = typename super_t::reference;
    using const_reference = typename super_t::const_reference;

    cudaStream_t stream{};

    ThrustCudaMallocAsyncAllocator(cudaStream_t stream_)
        : stream(stream_){
        
    }

    pointer allocate(size_type n){
        T* ptr = nullptr;
        cudaError_t status = cudaMallocAsync(&ptr, n * sizeof(T), stream);
        if(status != cudaSuccess){
            cudaGetLastError(); //reset error state
            std::cerr << "ThrustCudaMallocAsyncAllocator cuda error when allocating " << (n * sizeof(T)) << " bytes: " << cudaGetErrorString(status) << "\n";
            throw std::bad_alloc();
        }
        return thrust::device_pointer_cast(ptr);
    }

    void deallocate(pointer ptr, size_type /*n*/){
        cudaError_t status = cudaFreeAsync(ptr.get(), stream);
        if(status != cudaSuccess){
            cudaGetLastError(); //reset error state
            throw std::bad_alloc();
        }
    }
};

struct PairHmmSequenceGroup{
    int numHaps;
    int numReads;
    std::vector<uint8_t> readData;
    std::vector<uint8_t> hapData;
    std::vector<uint8_t> base_quals;
    std::vector<uint8_t> ins_quals;
    std::vector<uint8_t> del_quals;
    std::vector<uint8_t> gcp_quals;

    std::vector<int> readLengths;
    std::vector<int> hapLengths;
    std::vector<int> readBeginOffsets;
    std::vector<int> hapBeginOffsets;
};

struct BatchOfSequenceGroups{
public:
    friend class PinnedBatchOfSequenceGroups;

    template<class T>
    // using Allocator = std::allocator<T>;
    using Allocator = PinnedAllocator<T>;

    std::vector<uint8_t, Allocator<uint8_t>> readData;
    std::vector<uint8_t, Allocator<uint8_t>> hapData;

    std::vector<uint8_t, Allocator<uint8_t>> base_quals;
    std::vector<uint8_t, Allocator<uint8_t>> ins_quals;
    std::vector<uint8_t, Allocator<uint8_t>> del_quals;
    std::vector<uint8_t, Allocator<uint8_t>> gcp_quals;

    std::vector<int, Allocator<int>> numHapsPerSequenceGroup; 
    std::vector<int, Allocator<int>> numReadsPerSequenceGroup;

    std::vector<int, Allocator<int>> numHapsPerSequenceGroupPrefixSum;
    std::vector<int, Allocator<int>> numReadsPerSequenceGroupPrefixSum;

    std::vector<int, Allocator<int>> readLengths;
    std::vector<int, Allocator<int>> hapLengths;

    std::vector<int, Allocator<int>> readBeginOffsets;
    std::vector<int, Allocator<int>> hapBeginOffsets;

    int totalNumberOfAlignments;
    std::vector<int> numAlignmentsPerSequenceGroup;
    std::vector<int> numAlignmentsPerSequenceGroupPrefixSum;

    void reset(){
        readData.clear();
        hapData.clear();
        base_quals.clear();
        ins_quals.clear();
        del_quals.clear();
        gcp_quals.clear();
        numHapsPerSequenceGroup.clear();
        numReadsPerSequenceGroup.clear();
        numHapsPerSequenceGroupPrefixSum.clear();
        numReadsPerSequenceGroupPrefixSum.clear();
        readLengths.clear();
        hapLengths.clear();
        readBeginOffsets.clear();
        hapBeginOffsets.clear();
        totalNumberOfAlignments = 0;
        numAlignmentsPerSequenceGroup.clear();
        numAlignmentsPerSequenceGroupPrefixSum.clear();
    }

    bool operator==(const BatchOfSequenceGroups& rhs) const{
        if(readData != rhs.readData) return false;
        if(hapData != rhs.hapData) return false;
        if(base_quals != rhs.base_quals) return false;
        if(ins_quals != rhs.ins_quals) return false;
        if(del_quals != rhs.del_quals) return false;
        if(gcp_quals != rhs.gcp_quals) return false;
        if(numHapsPerSequenceGroup != rhs.numHapsPerSequenceGroup) return false;
        if(numReadsPerSequenceGroup != rhs.numReadsPerSequenceGroup) return false;
        if(numHapsPerSequenceGroupPrefixSum != rhs.numHapsPerSequenceGroupPrefixSum) return false;
        if(numReadsPerSequenceGroupPrefixSum != rhs.numReadsPerSequenceGroupPrefixSum) return false;
        if(readLengths != rhs.readLengths) return false;
        if(hapLengths != rhs.hapLengths) return false;
        if(readBeginOffsets != rhs.readBeginOffsets) return false;
        if(hapBeginOffsets != rhs.hapBeginOffsets) return false;
        if(numAlignmentsPerSequenceGroup != rhs.numAlignmentsPerSequenceGroup) return false;
        if(numAlignmentsPerSequenceGroupPrefixSum != rhs.numAlignmentsPerSequenceGroupPrefixSum) return false;
        return true;
    }
    bool operator!=(const BatchOfSequenceGroups& rhs) const{
        return !operator==(rhs);
    }

    int getTotalNumberOfAlignments() const{
        return totalNumberOfAlignments;
    }

    const std::vector<int>& getNumberOfAlignmentsPerBatch() const{
        return numAlignmentsPerSequenceGroup;
    }

    const std::vector<int>& getNumberOfAlignmentsPerBatchInclusivePrefixSum() const{
        return numAlignmentsPerSequenceGroupPrefixSum;
    }


private:



};

struct PinnedBatchOfSequenceGroups{
    template<class T>
    using Allocator = PinnedAllocator<T>;

    PinnedBatchOfSequenceGroups() = default;
    PinnedBatchOfSequenceGroups(const PinnedBatchOfSequenceGroups&) = default;
    PinnedBatchOfSequenceGroups& operator=(const PinnedBatchOfSequenceGroups&) = default;
    
    PinnedBatchOfSequenceGroups(const BatchOfSequenceGroups& rhs)
    : readData(rhs.readData.begin(), rhs.readData.end()),
     hapData(rhs.hapData.begin(), rhs.hapData.end()),
     base_quals(rhs.base_quals.begin(), rhs.base_quals.end()),
     ins_quals(rhs.ins_quals.begin(), rhs.ins_quals.end()),
     del_quals(rhs.del_quals.begin(), rhs.del_quals.end()),
     gcp_quals(rhs.gcp_quals.begin(), rhs.gcp_quals.end()),
     numHapsPerSequenceGroup(rhs.numHapsPerSequenceGroup.begin(), rhs.numHapsPerSequenceGroup.end()),
     numReadsPerSequenceGroup(rhs.numReadsPerSequenceGroup.begin(), rhs.numReadsPerSequenceGroup.end()),
     numHapsPerSequenceGroupPrefixSum(rhs.numHapsPerSequenceGroupPrefixSum.begin(), rhs.numHapsPerSequenceGroupPrefixSum.end()),
     numReadsPerSequenceGroupPrefixSum(rhs.numReadsPerSequenceGroupPrefixSum.begin(), rhs.numReadsPerSequenceGroupPrefixSum.end()),
     readLengths(rhs.readLengths.begin(), rhs.readLengths.end()),
     hapLengths(rhs.hapLengths.begin(), rhs.hapLengths.end()),
     readBeginOffsets(rhs.readBeginOffsets.begin(), rhs.readBeginOffsets.end()),
     hapBeginOffsets(rhs.hapBeginOffsets.begin(), rhs.hapBeginOffsets.end()),
     totalNumberOfAlignments(rhs.getTotalNumberOfAlignments()),
     numberOfAlignmentsPerBatch(rhs.getNumberOfAlignmentsPerBatch()),
     numAlignmentsPerBatchInclusivePrefixSum(rhs.getNumberOfAlignmentsPerBatchInclusivePrefixSum())
    {}

    std::vector<uint8_t, Allocator<uint8_t>> readData;             //all reads padded
    std::vector<uint8_t, Allocator<uint8_t>> hapData;              //all haplotypes padded

    std::vector<uint8_t, Allocator<uint8_t>> base_quals;        //base_qual - offset(33)
    std::vector<uint8_t, Allocator<uint8_t>> ins_quals;         //ins_qual - offset(33)
    std::vector<uint8_t, Allocator<uint8_t>> del_quals;         //del_qual - offset(33)
    std::vector<uint8_t, Allocator<uint8_t>> gcp_quals;         //gep_qual - offset(33)

    std::vector<int, Allocator<int>> numHapsPerSequenceGroup;
    std::vector<int, Allocator<int>> numReadsPerSequenceGroup;

    std::vector<int, Allocator<int>> numHapsPerSequenceGroupPrefixSum;
    std::vector<int, Allocator<int>> numReadsPerSequenceGroupPrefixSum;

    std::vector<int, Allocator<int>> readLengths;
    std::vector<int, Allocator<int>> hapLengths;

    std::vector<int, Allocator<int>> readBeginOffsets;
    std::vector<int, Allocator<int>> hapBeginOffsets;


    int getTotalNumberOfAlignments() const{
        return totalNumberOfAlignments;
    }

    const std::vector<int>& getNumberOfAlignmentsPerBatch() const{
        return numberOfAlignmentsPerBatch;
    }

    const std::vector<int>& getNumberOfAlignmentsPerBatchInclusivePrefixSum() const{
        return numAlignmentsPerBatchInclusivePrefixSum;
    }

private:
    std::vector<int> numberOfAlignmentsPerBatch;
    std::vector<int> numAlignmentsPerBatchInclusivePrefixSum;
    int totalNumberOfAlignments;    
};



template<class T>
struct SimpleConcurrentQueue{
    std::queue<T> queue;
    std::mutex mutex;
    std::condition_variable cv;

    void push(T item){
        std::lock_guard<std::mutex> lg(mutex);
        queue.emplace(std::move(item));
        cv.notify_one();
    }

    //wait until queue is not empty, then remove first element from queue and return it
    T pop(){
        std::unique_lock<std::mutex> ul(mutex);

        while(queue.empty()){
            cv.wait(ul);
        }

        T item = queue.front();
        queue.pop();
        return item;
    }
};




#endif