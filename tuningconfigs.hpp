#ifndef TUNINGCONFIGS_HPP
#define TUNINGCONFIGS_HPP



template<int blocksize_, int groupsize_, int numItems_>
struct AlignmentThreadLayout{
    static constexpr int blocksize = blocksize_;
    static constexpr int groupsize = groupsize_;
    static constexpr int numItems = numItems_;
    static constexpr int tileSize = groupsize * numItems;
    static constexpr int groupsPerBlock = blocksize / groupsize;
};

template<int blocksize_, int groupsize_, int numItems_>
struct PairHMMKernelConfig{
    using AlignmentThreadLayout = AlignmentThreadLayout<blocksize_, groupsize_, numItems_>;

    static constexpr int blocksize = AlignmentThreadLayout::blocksize;
    static constexpr int groupsize = AlignmentThreadLayout::groupsize;
    static constexpr int numItems = AlignmentThreadLayout::numItems;
    static constexpr int tileSize = AlignmentThreadLayout::tileSize;
    static constexpr int groupsPerBlock = AlignmentThreadLayout::groupsPerBlock;
};




template<int cudaArch>
struct PairHMMKernelConfigsSingleTile;

// "default" config for non-specialized archs
template<>
struct PairHMMKernelConfigsSingleTile<0>{
    using type = std::tuple<
        PairHMMKernelConfig<32, 4, 4>, // 16
        PairHMMKernelConfig<32, 4, 8>, // 32
        PairHMMKernelConfig<32, 4, 12>, // 48
        PairHMMKernelConfig<32, 4, 16>, // 64
        PairHMMKernelConfig<32, 4, 20>, // 80
        PairHMMKernelConfig<32, 4, 24>, // 96
        PairHMMKernelConfig<32, 4, 28>, // 112
        PairHMMKernelConfig<32, 4, 32>, // 128
        PairHMMKernelConfig<32, 8, 20>, // 160
        PairHMMKernelConfig<32, 8, 24>, // 192
        PairHMMKernelConfig<32, 8, 28>, // 224
        PairHMMKernelConfig<32, 8, 32>, // 256
        PairHMMKernelConfig<32, 16, 20>, // 320
        PairHMMKernelConfig<32, 16, 24>, // 384
        PairHMMKernelConfig<32, 16, 28>, // 448
        PairHMMKernelConfig<32, 16, 32>, // 512
        PairHMMKernelConfig<32, 32, 20>, // 640
        PairHMMKernelConfig<32, 32, 24>, // 768
        PairHMMKernelConfig<32, 32, 28>, // 896
        PairHMMKernelConfig<32, 32, 32> // 1024
    >;
};

template<>
struct PairHMMKernelConfigsSingleTile<700>{
    using type = std::tuple<
        PairHMMKernelConfig<32, 4, 4>, // 16
        PairHMMKernelConfig<32, 4, 8>, // 32
        PairHMMKernelConfig<32, 4, 12>, // 48
        PairHMMKernelConfig<32, 8, 8>, // 64
        PairHMMKernelConfig<32, 4, 20>, // 80
        PairHMMKernelConfig<32, 8, 12>, // 96
        PairHMMKernelConfig<32, 4, 28>, // 112
        PairHMMKernelConfig<32, 16, 8>, // 128
        PairHMMKernelConfig<32, 8, 20>, // 160
        PairHMMKernelConfig<32, 16, 12>, // 192
        PairHMMKernelConfig<32, 8, 28>, // 224
        PairHMMKernelConfig<32, 16, 16>, // 256
        PairHMMKernelConfig<32, 16, 20>, // 320
        PairHMMKernelConfig<32, 32, 12>, // 384
        PairHMMKernelConfig<32, 16, 28>, // 448
        PairHMMKernelConfig<32, 32, 16>, // 512
        PairHMMKernelConfig<32, 32, 20>, // 640
        PairHMMKernelConfig<32, 32, 24>, // 768
        PairHMMKernelConfig<32, 32, 28>, // 896
        PairHMMKernelConfig<32, 32, 32> // 1024
    >;
};

template<>
struct PairHMMKernelConfigsSingleTile<750>{
    using type = std::tuple<
        PairHMMKernelConfig<32, 4, 4>, // 16
        PairHMMKernelConfig<32, 4, 8>, // 32
        PairHMMKernelConfig<32, 4, 12>, // 48
        PairHMMKernelConfig<32, 4, 16>, // 64
        PairHMMKernelConfig<32, 4, 20>, // 80
        PairHMMKernelConfig<32, 4, 24>, // 96
        PairHMMKernelConfig<32, 4, 28>, // 112
        PairHMMKernelConfig<32, 8, 16>, // 128
        PairHMMKernelConfig<32, 8, 20>, // 160
        PairHMMKernelConfig<32, 16, 12>, // 192
        PairHMMKernelConfig<32, 8, 28>, // 224
        PairHMMKernelConfig<32, 16, 16>, // 256
        PairHMMKernelConfig<32, 16, 20>, // 320
        PairHMMKernelConfig<32, 16, 24>, // 384
        PairHMMKernelConfig<32, 16, 28>, // 448
        PairHMMKernelConfig<32, 32, 16>, // 512
        PairHMMKernelConfig<32, 32, 20>, // 640
        PairHMMKernelConfig<32, 32, 24>, // 768
        PairHMMKernelConfig<32, 32, 28>, // 896
        PairHMMKernelConfig<32, 32, 32> // 1024
    >;
};

template<>
struct PairHMMKernelConfigsSingleTile<800>{
    using type = std::tuple<
        PairHMMKernelConfig<32, 4, 4>, // 16
        PairHMMKernelConfig<32, 4, 8>, // 32
        PairHMMKernelConfig<32, 4, 12>, // 48
        PairHMMKernelConfig<32, 4, 16>, // 64
        PairHMMKernelConfig<32, 4, 20>, // 80
        PairHMMKernelConfig<32, 4, 24>, // 96
        PairHMMKernelConfig<32, 4, 28>, // 112
        PairHMMKernelConfig<32, 4, 32>, // 128
        PairHMMKernelConfig<32, 8, 20>, // 160
        PairHMMKernelConfig<32, 8, 24>, // 192
        PairHMMKernelConfig<32, 8, 28>, // 224
        PairHMMKernelConfig<32, 8, 32>, // 256
        PairHMMKernelConfig<32, 16, 20>, // 320
        PairHMMKernelConfig<32, 16, 24>, // 384
        PairHMMKernelConfig<32, 16, 28>, // 448
        PairHMMKernelConfig<32, 16, 32>, // 512
        PairHMMKernelConfig<32, 32, 20>, // 640
        PairHMMKernelConfig<32, 32, 24>, // 768
        PairHMMKernelConfig<32, 32, 28>, // 896
        PairHMMKernelConfig<32, 32, 32> // 1024
    >;
};

template<>
struct PairHMMKernelConfigsSingleTile<860>{
    using type = std::tuple<
        PairHMMKernelConfig<32, 4, 4>, // 16
        PairHMMKernelConfig<32, 4, 8>, // 32
        PairHMMKernelConfig<32, 4, 12>, // 48
        PairHMMKernelConfig<32, 4, 16>, // 64
        PairHMMKernelConfig<32, 4, 20>, // 80
        PairHMMKernelConfig<32, 4, 24>, // 96
        PairHMMKernelConfig<32, 4, 28>, // 112
        PairHMMKernelConfig<32, 8, 16>, // 128
        PairHMMKernelConfig<32, 8, 20>, // 160
        PairHMMKernelConfig<32, 8, 24>, // 192
        PairHMMKernelConfig<32, 8, 28>, // 224
        PairHMMKernelConfig<32, 16, 16>, // 256
        PairHMMKernelConfig<32, 16, 20>, // 320
        PairHMMKernelConfig<32, 16, 24>, // 384
        PairHMMKernelConfig<32, 16, 28>, // 448
        PairHMMKernelConfig<32, 32, 16>, // 512
        PairHMMKernelConfig<32, 32, 20>, // 640
        PairHMMKernelConfig<32, 32, 24>, // 768
        PairHMMKernelConfig<32, 32, 28>, // 896
        PairHMMKernelConfig<32, 32, 32> // 1024
    >;
};

template<>
struct PairHMMKernelConfigsSingleTile<890>{
    using type = std::tuple<
        PairHMMKernelConfig<32, 4, 4>, // 16
        PairHMMKernelConfig<32, 4, 8>, // 32
        PairHMMKernelConfig<32, 4, 12>, // 48
        PairHMMKernelConfig<32, 4, 16>, // 64
        PairHMMKernelConfig<32, 4, 20>, // 80
        PairHMMKernelConfig<32, 8, 12>, // 96
        PairHMMKernelConfig<32, 4, 28>, // 112
        PairHMMKernelConfig<32, 8, 16>, // 128
        PairHMMKernelConfig<32, 8, 20>, // 160
        PairHMMKernelConfig<32, 16, 12>, // 192
        PairHMMKernelConfig<32, 8, 28>, // 224
        PairHMMKernelConfig<32, 16, 16>, // 256
        PairHMMKernelConfig<32, 16, 20>, // 320
        PairHMMKernelConfig<32, 32, 12>, // 384
        PairHMMKernelConfig<32, 16, 28>, // 448
        PairHMMKernelConfig<32, 32, 16>, // 512
        PairHMMKernelConfig<32, 32, 20>, // 640
        PairHMMKernelConfig<32, 32, 24>, // 768
        PairHMMKernelConfig<32, 32, 28>, // 896
        PairHMMKernelConfig<32, 32, 32> // 1024
    >;
};

template<>
struct PairHMMKernelConfigsSingleTile<900>{
    using type = std::tuple<
        PairHMMKernelConfig<32, 4, 4>, // 16
        PairHMMKernelConfig<32, 4, 8>, // 32
        PairHMMKernelConfig<32, 4, 12>, // 48
        PairHMMKernelConfig<32, 4, 16>, // 64
        PairHMMKernelConfig<32, 4, 20>, // 80
        PairHMMKernelConfig<32, 8, 12>, // 96
        PairHMMKernelConfig<32, 4, 28>, // 112
        PairHMMKernelConfig<32, 4, 32>, // 128
        PairHMMKernelConfig<32, 8, 20>, // 160
        PairHMMKernelConfig<32, 8, 24>, // 192
        PairHMMKernelConfig<32, 8, 28>, // 224
        PairHMMKernelConfig<32, 8, 32>, // 256
        PairHMMKernelConfig<32, 16, 20>, // 320
        PairHMMKernelConfig<32, 16, 24>, // 384
        PairHMMKernelConfig<32, 16, 28>, // 448
        PairHMMKernelConfig<32, 16, 32>, // 512
        PairHMMKernelConfig<32, 32, 20>, // 640
        PairHMMKernelConfig<32, 32, 24>, // 768
        PairHMMKernelConfig<32, 32, 28>, // 896
        PairHMMKernelConfig<32, 32, 32> // 1024
    >;
};

template<>
struct PairHMMKernelConfigsSingleTile<1200>{
    using type = std::tuple<
        PairHMMKernelConfig<32, 4, 4>, // 16
        PairHMMKernelConfig<32, 4, 8>, // 32
        PairHMMKernelConfig<32, 4, 12>, // 48
        PairHMMKernelConfig<32, 4, 16>, // 64
        PairHMMKernelConfig<32, 4, 20>, // 80
        PairHMMKernelConfig<32, 8, 12>, // 96
        PairHMMKernelConfig<32, 4, 28>, // 112
        PairHMMKernelConfig<32, 8, 16>, // 128
        PairHMMKernelConfig<32, 8, 20>, // 160
        PairHMMKernelConfig<32, 16, 12>, // 192
        PairHMMKernelConfig<32, 8, 28>, // 224
        PairHMMKernelConfig<32, 16, 16>, // 256
        PairHMMKernelConfig<32, 16, 20>, // 320
        PairHMMKernelConfig<32, 32, 12>, // 384
        PairHMMKernelConfig<32, 16, 28>, // 448
        PairHMMKernelConfig<32, 32, 16>, // 512
        PairHMMKernelConfig<32, 32, 20>, // 640
        PairHMMKernelConfig<32, 32, 24>, // 768
        PairHMMKernelConfig<32, 32, 28>, // 896
        PairHMMKernelConfig<32, 32, 32> // 1024
    >;
};


template <class T, class S = void>
struct TypeExists : std::false_type {};

template <class T>
struct TypeExists<T, std::void_t<decltype(sizeof(T) != 0)>> : std::true_type {};

template<
    int cudaArch,
    template<int> class RealConfig,
    class S = void
>
struct SelectRealConfigOrCompatibilityConfig{
    static constexpr int arch = 0;
    using type = PairHMMKernelConfigsSingleTile<0>;
};

template<
    int cudaArch,
    template<int> class RealConfig
>
struct SelectRealConfigOrCompatibilityConfig<
    cudaArch,
    RealConfig,
    typename std::enable_if<TypeExists<RealConfig<cudaArch>>::value>::type
>{
    static constexpr int arch = cudaArch;
    using type = RealConfig<cudaArch>;
};


#endif