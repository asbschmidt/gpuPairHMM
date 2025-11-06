#ifndef PAIRHMM_KERNELS_CUH
#define PAIRHMM_KERNELS_CUH

#include <cuda/std/cstdint>
#include <cuda/std/limits>

#include <thrust/binary_search.h>

#ifndef SDIV
#define SDIV(x,y)(((x)+(y)-1)/(y))
#endif



__constant__ float cPH2PR[128];


template <int group_size, int numRegs> 
__global__
void PairHMM_align_partition_float_allowMultipleSequenceGroupsPerWarp(
    float* resultoutput,
    const uint8_t* readData,
    const uint8_t* hapData,
    const uint8_t* base_quals,
    const uint8_t* ins_quals,
    const uint8_t* del_quals,
    const int* readBeginOffsets,
    const int* hapBeginOffsets,
    const int* readLengths,
    const int* hapLengths,
    const int* numHapsPerSequenceGroup,
    const int* numHapsPerSequenceGroupPrefixSum,
    const int* indicesPerSequenceGroup,
    const int* numReadsPerSequenceGroupPrefixSum,
    const int numSequenceGroups,
    const int* resultOffsetsPerSequenceGroup,

    const int* numAlignmentsPerSequenceGroup,
    const int* numAlignmentsPerSequenceGroupInclusivePrefixSum,
    const int numAlignments
) {

    alignas(8) __shared__ float2 lambda_array[5][16*numRegs];

    float M[numRegs], I, D[numRegs];
    float alpha[numRegs], delta[numRegs], sigma[numRegs];
    float Results[numRegs];

    const int threadIdInGroup = threadIdx.x % group_size;
    const int threadGroupIdInBlock = threadIdx.x / group_size;
    const int threadGroupIdInGrid = (threadIdx.x + blockIdx.x * blockDim.x) / group_size;
    const unsigned int myGroupMask = __match_any_sync(0xFFFFFFFF, threadGroupIdInGrid); //compute mask for all threads with same threadGroupIdInGrid
    
    // const int numGroupsInGrid = blockDim.x * gridDim.x / group_size;
    // for(int alignmentId = threadGroupIdInGrid; alignmentId < numAlignments; alignmentId += numGroupsInGrid){
    const int alignmentId = threadGroupIdInGrid;
    if(alignmentId < numAlignments){

        const int sequenceGroupIdByThreadGroupId = thrust::distance(
            numAlignmentsPerSequenceGroupInclusivePrefixSum,
            thrust::upper_bound(thrust::seq,
                numAlignmentsPerSequenceGroupInclusivePrefixSum,
                numAlignmentsPerSequenceGroupInclusivePrefixSum + numSequenceGroups,
                alignmentId
            )
        );
        const int sequenceGroupId = min(sequenceGroupIdByThreadGroupId, numSequenceGroups-1);
        const int threadGroupIdInSequenceGroup = alignmentId - (sequenceGroupId == 0 ? 0 : numAlignmentsPerSequenceGroupInclusivePrefixSum[sequenceGroupId-1]);
        const int hapToProcessInSequenceGroup = threadGroupIdInSequenceGroup % numHapsPerSequenceGroup[sequenceGroupId];
        const int readIndexToProcessInSequenceGroup = threadGroupIdInSequenceGroup / numHapsPerSequenceGroup[sequenceGroupId];

        const int readIndexOffset = numReadsPerSequenceGroupPrefixSum[sequenceGroupId];
        const int readIndexOffset_inChunk = readIndexOffset - numReadsPerSequenceGroupPrefixSum[0];
        const int readToProcessInSequenceGroup = indicesPerSequenceGroup[readIndexOffset_inChunk + readIndexToProcessInSequenceGroup];

        const int read_nr = readToProcessInSequenceGroup;
        // const int global_read_id = read_nr + readIndexOffset;
        const int read_id_inChunk = read_nr + readIndexOffset_inChunk;


        const int byteOffsetForRead = readBeginOffsets[read_id_inChunk];
        const int readLength = readLengths[read_id_inChunk];

        const int b_h_off = numHapsPerSequenceGroupPrefixSum[sequenceGroupId];
        const int b_h_off_inChunk = b_h_off - numHapsPerSequenceGroupPrefixSum[0];
        const int bytesOffsetForHap = hapBeginOffsets[hapToProcessInSequenceGroup+b_h_off_inChunk];
        const char4* const HapsAsChar4 = reinterpret_cast<const char4*>(&hapData[bytesOffsetForHap]);
        const int haploLength = hapLengths[hapToProcessInSequenceGroup+b_h_off_inChunk];

        const int resultOutputIndex = resultOffsetsPerSequenceGroup[sequenceGroupId] + read_nr*numHapsPerSequenceGroup[sequenceGroupId]+hapToProcessInSequenceGroup;
        // if(threadIdInGroup == 0){
        //     printf("group %d, sequenceGroupIdByThreadGroupId %d, sequenceGroupId %d, threadGroupIdInSequenceGroup %d, hapToProcessInSequenceGroup %d, readIndexToProcessInSequenceGroup %d, readToProcessInSequenceGroup %d, numAlignments %d\n"
        //         "resultOffsetsPerSequenceGroup %d, numHapsPerSequenceGroup %d, resultOutputIndex %d\n",
        //         alignmentId, sequenceGroupIdByThreadGroupId, sequenceGroupId, threadGroupIdInSequenceGroup, hapToProcessInSequenceGroup, readIndexToProcessInSequenceGroup, readToProcessInSequenceGroup, numAlignments,
        //         resultOffsetsPerSequenceGroup[sequenceGroupId], numHapsPerSequenceGroup[sequenceGroupId], resultOutputIndex);
        // }


        // if(alignmentId < 10 && group_size == 8 && numRegs == 8){
            // if(threadIdInGroup == 0){
            //     printf("group %d, sequenceGroupId %d, threadGroupIdInSequenceGroup %d, hapToProcessInSequenceGroup %d, readIndexToProcessInSequenceGroup %d, readToProcessInSequenceGroup %d, readLength %d, haploLength %d, numAlignments %d\n"
            //         "readIndexOffset %d, readToProcessInSequenceGroup %d, byteOffsetForRead %d, b_h_off %d, bytesOffsetForHap %d\n"
            //         "resultOffsetsPerSequenceGroup %d, numHapsPerSequenceGroup %d\n",
            //         alignmentId, sequenceGroupId, threadGroupIdInSequenceGroup, hapToProcessInSequenceGroup, readIndexToProcessInSequenceGroup, readToProcessInSequenceGroup, readLength, haploLength, numAlignments,
            //         readIndexOffset, readToProcessInSequenceGroup, byteOffsetForRead, b_h_off, bytesOffsetForHap,
            //         resultOffsetsPerSequenceGroup[sequenceGroupId], numHapsPerSequenceGroup[sequenceGroupId]);
            // }
        // }

        const float eps = 0.1;
        const float beta = 0.9;
        float M_l, D_l, M_ul, D_ul, I_ul;
        float penalty_temp0, penalty_temp1, penalty_temp2, penalty_temp3;
        float init_D;

        const float constant = ::cuda::std::numeric_limits<float>::max() / 16;

        auto load_PSSM = [&]() {

            char4 temp0, temp1;
            const float one = 1.0;
            const float three = 3.0;
            const char4* QualsAsChar4 = reinterpret_cast<const char4*>(&base_quals[byteOffsetForRead]);
            const char4* ReadsAsChar4 = reinterpret_cast<const char4*>(&readData[byteOffsetForRead]);
            for (int i=threadIdInGroup; i<(readLength+3)/4; i+=group_size) {
                float2 temp_h2, temp_h3;
                temp0 = QualsAsChar4[i];
                temp1 = ReadsAsChar4[i];
                temp_h2.x = cPH2PR[uint8_t(temp0.x)];
                temp_h2.y = cPH2PR[uint8_t(temp0.y)];
                temp_h3.x = temp_h2.x/three;
                temp_h3.y = temp_h2.y/three;

                //init hap == A,C,G,T as mismatch
                for (int j=0; j<4; j++){
                    lambda_array[j][2*i+threadGroupIdInBlock*(group_size*numRegs/2)] = temp_h3; // mismatch
                }
                //hap == N always matches
                lambda_array[4][2*i+threadGroupIdInBlock*(group_size*numRegs/2)].x = one - temp_h2.x; // match
                lambda_array[4][2*i+threadGroupIdInBlock*(group_size*numRegs/2)].y = one - temp_h2.y; // match

                if (temp1.x < 4){
                    // set hap == read
                    lambda_array[temp1.x][2*i+threadGroupIdInBlock*(group_size*numRegs/2)].x = one - temp_h2.x; // match
                }else if (temp1.x == 4){
                    // read == N always matches
                    for (int j=0; j<4; j++){
                        lambda_array[j][2*i+threadGroupIdInBlock*(group_size*numRegs/2)].x = one - temp_h2.x; // N always match
                    }
                }
                if (temp1.y < 4){
                    // set hap == read
                    lambda_array[temp1.y][2*i+threadGroupIdInBlock*(group_size*numRegs/2)].y = one - temp_h2.y; // match
                }else if (temp1.y == 4){
                    // read == N always matches
                    for (int j=0; j<4; j++){
                        lambda_array[j][2*i+threadGroupIdInBlock*(group_size*numRegs/2)].y = one - temp_h2.y; // N always match
                    }
                }

                temp_h2.x = cPH2PR[uint8_t(temp0.z)];
                temp_h2.y = cPH2PR[uint8_t(temp0.w)];
                temp_h3.x = temp_h2.x/three;
                temp_h3.y = temp_h2.y/three;

                //init hap == A,C,G,T as mismatch
                for (int j=0; j<4; j++){
                    lambda_array[j][2*i+1+threadGroupIdInBlock*(group_size*numRegs/2)] = temp_h3; // mismatch
                }
                //hap == N always matches
                lambda_array[4][2*i+1+threadGroupIdInBlock*(group_size*numRegs/2)].x = one - temp_h2.x; // match
                lambda_array[4][2*i+1+threadGroupIdInBlock*(group_size*numRegs/2)].y = one - temp_h2.y; // match

                if (temp1.z < 4){
                    // set hap == read
                    lambda_array[temp1.z][2*i+1+threadGroupIdInBlock*(group_size*numRegs/2)].x = one - temp_h2.x; // match
                }else if (temp1.z == 4){
                    // read == N always matches
                    for (int j=0; j<4; j++){
                        lambda_array[j][2*i+1+threadGroupIdInBlock*(group_size*numRegs/2)].x = one - temp_h2.x; // N always match
                    }
                }
                if (temp1.w < 4){
                    // set hap == read
                    lambda_array[temp1.w][2*i+1+threadGroupIdInBlock*(group_size*numRegs/2)].y = one - temp_h2.y; // match
                }else if (temp1.w == 4){
                    // read == N always matches
                    for (int j=0; j<4; j++){
                        lambda_array[j][2*i+1+threadGroupIdInBlock*(group_size*numRegs/2)].y = one - temp_h2.y; // N always match
                    }
                }
            }

            __syncwarp(myGroupMask);

            // if(threadIdInGroup == 0){
            //     printf("float kernel pssm\n");
            //     for(int r = 0; r < 5; r++){
            //         for(int c = 0; c < 16*numRegs; c++){
            //             printf("%f %f ", lambda_array[r][c].x, lambda_array[r][c].y);
            //         }
            //         printf("\n");
            //     }
            // }

        };

        auto load_probabilities = [&]() {
            char4 temp0, temp1;
            const char4* InsQualsAsChar4 = reinterpret_cast<const char4*>(&ins_quals[byteOffsetForRead]);
            const char4* DelQualsAsChar4 = reinterpret_cast<const char4*>(&del_quals[byteOffsetForRead]);
            for (int i=0; i<numRegs/4; i++) {
                if (threadIdInGroup*numRegs/4+i < (readLength+3)/4) {

                    temp0 = InsQualsAsChar4[threadIdInGroup*numRegs/4+i];
                    temp1 = DelQualsAsChar4[threadIdInGroup*numRegs/4+i];

                    delta[4*i] = cPH2PR[uint8_t(temp0.x)];
                    delta[4*i+1] = cPH2PR[uint8_t(temp0.y)];
                    delta[4*i+2] = cPH2PR[uint8_t(temp0.z)];
                    delta[4*i+3] = cPH2PR[uint8_t(temp0.w)];

                    sigma[4*i] = cPH2PR[uint8_t(temp1.x)];
                    sigma[4*i+1] = cPH2PR[uint8_t(temp1.y)];
                    sigma[4*i+2] = cPH2PR[uint8_t(temp1.z)];
                    sigma[4*i+3] = cPH2PR[uint8_t(temp1.w)];

                    alpha[4*i] = 1.0 - (delta[4*i] + sigma[4*i]);
                    alpha[4*i+1] = 1.0 - (delta[4*i+1] + sigma[4*i+1]);
                    alpha[4*i+2] = 1.0 - (delta[4*i+2] + sigma[4*i+2]);
                    alpha[4*i+3] = 1.0 - (delta[4*i+3] + sigma[4*i+3]);
                }
            }

        };

        auto init_penalties = [&]() {
            #pragma unroll
            for (int i=0; i<numRegs; i++) M[i] = D[i] = Results[i] = 0.0;
            M_l = M_ul = D_ul = I_ul = D_l = I = 0.0;
            if (!threadIdInGroup) D_l = D_ul = init_D;
        };


        char hap_letter;

        
        auto calc_DP_float = [&](int row){
            

            float2* sbt_row = lambda_array[hap_letter];
            float4 lambda = *((float4*)&sbt_row[threadIdx.x*numRegs/2]);

            penalty_temp0 = M[0];
            penalty_temp1 = D[0];
            M[0] = lambda.x * fmaf(alpha[0],M_ul,beta*(I_ul+D_ul));
            D[0] = fmaf(sigma[0],penalty_temp0,eps*D[0]);
            I = fmaf(delta[0],M_ul,eps*I_ul);
            Results[0] += M[0] + I;
            penalty_temp2 = M[1];
            penalty_temp3 = D[1];
            M[1] = lambda.y * fmaf(alpha[1],penalty_temp0,beta*(I+penalty_temp1));
            D[1] = fmaf(sigma[1],penalty_temp2,eps*D[1]);
            I = fmaf(delta[1],penalty_temp0,eps*I);
            Results[1] += M[1] + I;


            penalty_temp0 = M[2];
            penalty_temp1 = D[2];
            M[2] = lambda.z * fmaf(alpha[2],penalty_temp2,beta*(I+penalty_temp3));
            D[2] = fmaf(sigma[2],penalty_temp0,eps*D[2]);
            I = fmaf(delta[2],penalty_temp2,eps*I);
            Results[2] += M[2] + I;
            penalty_temp2 = M[3];
            penalty_temp3 = D[3];
            M[3] = lambda.w * fmaf(alpha[3],penalty_temp0,beta*(I+penalty_temp1));
            D[3] = fmaf(sigma[3],penalty_temp2,eps*D[3]);
            I = fmaf(delta[3],penalty_temp0,eps*I);
            Results[3] += M[3] + I;

            // if(threadIdInGroup * numRegs + 0 < haploLength) printf("row %d, col %d, lambda %f\n", row, threadIdInGroup * numRegs + 0, lambda.x);
            // if(threadIdInGroup * numRegs + 1 < haploLength) printf("row %d, col %d, lambda %f\n", row, threadIdInGroup * numRegs + 1, lambda.y);
            // if(threadIdInGroup * numRegs + 2 < haploLength) printf("row %d, col %d, lambda %f\n", row, threadIdInGroup * numRegs + 2, lambda.z);
            // if(threadIdInGroup * numRegs + 3 < haploLength) printf("row %d, col %d, lambda %f\n", row, threadIdInGroup * numRegs + 3, lambda.w);

            #pragma unroll
            for (int i=1; i<numRegs/4; i++) {
                float4 lambda = *((float4*)&sbt_row[threadIdx.x*numRegs/2+2*i]);
                //memcpy(&score2, &lambda.x, sizeof(float2));
                penalty_temp0 = M[4*i];
                penalty_temp1 = D[4*i];
                M[4*i] = lambda.x * fmaf(alpha[4*i],penalty_temp2,beta*(I+penalty_temp3));
                D[4*i] = fmaf(sigma[4*i],penalty_temp0,eps*D[4*i]);
                I = fmaf(delta[4*i],penalty_temp2,eps*I);
                Results[4*i] += M[4*i] + I;
                penalty_temp2 = M[4*i+1];
                penalty_temp3 = D[4*i+1];
                M[4*i+1] = lambda.y * fmaf(alpha[4*i+1],penalty_temp0,beta*(I+penalty_temp1));
                D[4*i+1] = fmaf(sigma[4*i+1],penalty_temp2,eps*D[4*i+1]);
                I = fmaf(delta[4*i+1],penalty_temp0,eps*I);
                Results[4*i+1] += M[4*i+1] + I;

                //memcpy(&score2, &lambda.z, sizeof(float2));
                penalty_temp0 = M[4*i+2];
                penalty_temp1 = D[4*i+2];
                M[4*i+2] = lambda.z * fmaf(alpha[4*i+2],penalty_temp2,beta*(I+penalty_temp3));
                D[4*i+2] = fmaf(sigma[4*i+2],penalty_temp0,eps*D[4*i+2]);
                I = fmaf(delta[4*i+2],penalty_temp2,eps*I);
                Results[4*i+2] += M[4*i+2] + I;

                penalty_temp2 = M[4*i+3];
                penalty_temp3 = D[4*i+3];
                M[4*i+3] = lambda.w * fmaf(alpha[4*i+3],penalty_temp0,beta*(I+penalty_temp1));
                D[4*i+3] = fmaf(sigma[4*i+3],penalty_temp2,eps*D[4*i+3]);
                I = fmaf(delta[4*i+3],penalty_temp0,eps*I);
                Results[4*i+3] += M[4*i+3] + I;

            //     if(threadIdInGroup * numRegs + 4*i + 0 < haploLength) printf("row %d, col %d, lambda %f\n", row, threadIdInGroup * numRegs + 4*i + 0, lambda.x);
            //     if(threadIdInGroup * numRegs + 4*i + 1 < haploLength) printf("row %d, col %d, lambda %f\n", row, threadIdInGroup * numRegs + 4*i + 1, lambda.y);
            //     if(threadIdInGroup * numRegs + 4*i + 2 < haploLength) printf("row %d, col %d, lambda %f\n", row, threadIdInGroup * numRegs + 4*i + 2, lambda.z);
            //     if(threadIdInGroup * numRegs + 4*i + 3 < haploLength) printf("row %d, col %d, lambda %f\n", row, threadIdInGroup * numRegs + 4*i + 3, lambda.w);
            }
        };

        auto shuffle_penalty = [&]() {
            M_ul = M_l;
            D_ul = D_l;

            M_l = __shfl_up_sync(myGroupMask, M[numRegs-1], 1, group_size);
            I_ul = __shfl_up_sync(myGroupMask, I, 1, group_size);
            D_l = __shfl_up_sync(myGroupMask, D[numRegs-1], 1, group_size);

            if (!threadIdInGroup) {
                M_l = I_ul = 0.0;
                D_l = init_D;
            }
        };

        int result_thread = (readLength-1)/numRegs;
        int result_reg = (readLength-1)%numRegs;

        load_PSSM();
        load_probabilities();
        // compute_probabilities();

        init_D = constant/haploLength;
        init_penalties();

        char4 new_hap_letter4;
        hap_letter = 4;
        int k;
        for (k=0; k<haploLength-3; k+=4) {
            new_hap_letter4 = HapsAsChar4[k/4];
            hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
            if (!threadIdInGroup) hap_letter = new_hap_letter4.x;
            calc_DP_float(k);
            shuffle_penalty();

            hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
            if (!threadIdInGroup) hap_letter = new_hap_letter4.y;
            calc_DP_float(k+1);
            shuffle_penalty();

            hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
            if (!threadIdInGroup) hap_letter = new_hap_letter4.z;
            calc_DP_float(k+2);
            shuffle_penalty();

            hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
            if (!threadIdInGroup) hap_letter = new_hap_letter4.w;
            calc_DP_float(k+3);
            shuffle_penalty();
        }
        if (haploLength%4 >= 1) {
            new_hap_letter4 = HapsAsChar4[k/4];
            hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
            if (!threadIdInGroup) hap_letter = new_hap_letter4.x;
            calc_DP_float(k);
            shuffle_penalty();
        }
        if (haploLength%4 >= 2) {
            hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
            if (!threadIdInGroup) hap_letter = new_hap_letter4.y;
            calc_DP_float(k+1);
            shuffle_penalty();
        }
        if (haploLength%4 >= 3) {
            hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
            if (!threadIdInGroup) hap_letter = new_hap_letter4.z;
            calc_DP_float(k+2);
            shuffle_penalty();
        }
        for (k=0; k<result_thread; k++) {
            // hap_letter = __shfl_up_sync(__activemask(), hap_letter, 1, 32);
            hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
            calc_DP_float(haploLength+k);
            shuffle_penalty(); // shuffle_penalty_active();
        }
        // adjust I values
        I = fmaf(delta[0],M_ul,eps*I_ul);
        Results[0] += I;
        I = fmaf(delta[1],M[0],eps*I);
        Results[1] += I;
        for (int p=1; p<numRegs/2; p++) {
            I = fmaf(delta[2*p],M[2*p-1],eps*I);
            Results[2*p] += I;
            I = fmaf(delta[2*p+1],M[2*p],eps*I);
            Results[2*p+1] += I;
        }
        // adjust I values
        //I = fmaf(delta[0],M_ul,eps*I_ul);
        //Results[0] += I;
        //for (int p=1; p<numRegs; p++) {
        //    I = fmaf(delta[p],M[p-1],eps*I);
        //    Results[p] += I;
        //}


        if (threadIdInGroup == result_thread) {
            float temp_res = Results[result_reg];
            temp_res =  log10f(temp_res) - log10f(constant);
            resultoutput[resultOutputIndex] = temp_res;
        }
    }


}


template <int group_size, int numRegs> 
__global__
void PairHMM_align_partition_float_allowMultipleSequenceGroupsPerWarp_coalesced_smem_noimprovedResultComputation(
    float* resultoutput,
    const uint8_t* readData,
    const uint8_t* hapData,
    const uint8_t* base_quals,
    const uint8_t* ins_quals,
    const uint8_t* del_quals,
    const int* readBeginOffsets,
    const int* hapBeginOffsets,
    const int* readLengths,
    const int* hapLengths,
    const int* numHapsPerSequenceGroup,
    const int* numHapsPerSequenceGroupPrefixSum,
    const int* indicesPerSequenceGroup,
    const int* numReadsPerSequenceGroupPrefixSum,
    const int numSequenceGroups,
    const int* resultOffsetsPerSequenceGroup,
    const int* numAlignmentsPerSequenceGroup,
    const int* numAlignmentsPerSequenceGroupInclusivePrefixSum,
    const int numAlignments
) {
    static_assert(numRegs % 4 == 0);

    constexpr int warpsize = 32;
    constexpr int blocksize = 32;
    constexpr int numGroupsPerBlock = blocksize / group_size;

    constexpr int rowsize = numGroupsPerBlock * group_size*numRegs;
    alignas(16) __shared__ float lambda_array_permuted[5][rowsize];

    float M[numRegs], I, D[numRegs];
    float alpha[numRegs], delta[numRegs], sigma[numRegs];
    alignas(16) float Results[numRegs];

    const int threadIdInGroup = threadIdx.x % group_size;
    // const int threadGroupIdInBlock = threadIdx.x / group_size;
    const int threadIdInWarp = threadIdx.x % warpsize;
    const int threadGroupIdInWarp = threadIdInWarp / group_size;
    const int threadGroupIdInGrid = (threadIdx.x + blockIdx.x * blockDim.x) / group_size;
    const unsigned int myGroupMask = __match_any_sync(0xFFFFFFFF, threadGroupIdInGrid); //compute mask for all threads with same threadGroupIdInGrid
    
    // const int numGroupsInGrid = blockDim.x * gridDim.x / group_size;
    // for(int alignmentId = threadGroupIdInGrid; alignmentId < numAlignments; alignmentId += numGroupsInGrid){
    const int alignmentId = threadGroupIdInGrid;
    if(alignmentId < numAlignments){

        const int sequenceGroupIdByThreadGroupId = thrust::distance(
            numAlignmentsPerSequenceGroupInclusivePrefixSum,
            thrust::upper_bound(thrust::seq,
                numAlignmentsPerSequenceGroupInclusivePrefixSum,
                numAlignmentsPerSequenceGroupInclusivePrefixSum + numSequenceGroups,
                alignmentId
            )
        );
        const int sequenceGroupId = min(sequenceGroupIdByThreadGroupId, numSequenceGroups-1);
        const int threadGroupIdInSequenceGroup = alignmentId - (sequenceGroupId == 0 ? 0 : numAlignmentsPerSequenceGroupInclusivePrefixSum[sequenceGroupId-1]);
        const int hapToProcessInSequenceGroup = threadGroupIdInSequenceGroup % numHapsPerSequenceGroup[sequenceGroupId];
        const int readIndexToProcessInSequenceGroup = threadGroupIdInSequenceGroup / numHapsPerSequenceGroup[sequenceGroupId];

        const int readIndexOffset = numReadsPerSequenceGroupPrefixSum[sequenceGroupId];
        const int readIndexOffset_inChunk = readIndexOffset - numReadsPerSequenceGroupPrefixSum[0];
        const int readToProcessInSequenceGroup = indicesPerSequenceGroup[readIndexOffset_inChunk + readIndexToProcessInSequenceGroup];

        const int read_nr = readToProcessInSequenceGroup;
        // const int global_read_id = read_nr + readIndexOffset;
        const int read_id_inChunk = read_nr + readIndexOffset_inChunk;


        const int byteOffsetForRead = readBeginOffsets[read_id_inChunk];
        const int readLength = readLengths[read_id_inChunk];

        const int b_h_off = numHapsPerSequenceGroupPrefixSum[sequenceGroupId];
        const int b_h_off_inChunk = b_h_off - numHapsPerSequenceGroupPrefixSum[0];
        const int bytesOffsetForHap = hapBeginOffsets[hapToProcessInSequenceGroup+b_h_off_inChunk];
        const char4* const HapsAsChar4 = reinterpret_cast<const char4*>(&hapData[bytesOffsetForHap]);
        const int haploLength = hapLengths[hapToProcessInSequenceGroup+b_h_off_inChunk];

        const int resultOutputIndex = resultOffsetsPerSequenceGroup[sequenceGroupId] + read_nr*numHapsPerSequenceGroup[sequenceGroupId]+hapToProcessInSequenceGroup;

        // if(threadIdx.x == 0){
        //     printf("old kernel\n");

        //     printf("cPH2PR\n");
        //     for(int i = 0; i < 128; i++){
        //         printf("%f ", (cPH2PR[i]));
        //     }
        //     printf("\n");

        //     printf("read\n");
        //     for(int i = 0; i < readLength; i++){
        //         printf("%d", int(readData[byteOffsetForRead + i]));
        //     }
        //     printf("\n");
        //     printf("base\n");
        //     for(int i = 0; i < readLength; i++){
        //         printf("%c", (base_quals[byteOffsetForRead + i])+33);
        //     }
        //     printf("\n");
        //     printf("ins\n");
        //     for(int i = 0; i < readLength; i++){
        //         printf("%c", (ins_quals[byteOffsetForRead + i])+33);
        //     }
        //     printf("\n");
        //     printf("del\n");
        //     for(int i = 0; i < readLength; i++){
        //         printf("%c", (del_quals[byteOffsetForRead + i])+33);
        //     }
        //     printf("\n");
        //     printf("hap\n");
        //     for(int i = 0; i < haploLength; i++){
        //         printf("%c", (hapData[bytesOffsetForHap + i]));
        //     }
        //     printf("\n");
        // }

        const float eps = 0.1;
        const float beta = 0.9;
        float M_l, D_l, M_ul, D_ul, I_ul;
        float penalty_temp0, penalty_temp1, penalty_temp2, penalty_temp3;
        float init_D;

        const float constant = ::cuda::std::numeric_limits<float>::max() / 16;

        int result_thread = (readLength-1)/numRegs;
        int result_reg = (readLength-1)%numRegs;

        auto construct_PSSM_warp_coalesced = [&](){
            __syncwarp(myGroupMask);
            
            const char4* QualsAsChar4 = reinterpret_cast<const char4*>(&base_quals[byteOffsetForRead]);
            const char4* ReadsAsChar4 = reinterpret_cast<const char4*>(&readData[byteOffsetForRead]);
            for (int i=threadIdInGroup; i<(readLength+3)/4; i+=group_size) {
                const char4 temp0 = QualsAsChar4[i];
                const char4 temp1 = ReadsAsChar4[i];
                alignas(4) char quals[4];
                memcpy(&quals[0], &temp0, sizeof(char4));
                alignas(4) char letters[4];
                memcpy(&letters[0], &temp1, sizeof(char4));

                float probs[4];
                #pragma unroll
                for(int c = 0; c < 4; c++){
                    probs[c] = cPH2PR[quals[c]];
                }

                alignas(16) float rowResult[5][4];

                #pragma unroll
                for(int c = 0; c < 4; c++){
                    //hap == N always matches
                    rowResult[4][c] = 1 - probs[c]; //match

                    if(letters[c] < 4){
                        // set hap == read to 1 - prob, hap != read to prob / 3
                        #pragma unroll
                        for (int j=0; j<4; j++){
                            rowResult[j][c] = (j == letters[c]) ? 1 - probs[c] : probs[c]/3.0f; //match or mismatch
                        }
                    }else{
                        // read == N always matches
                        #pragma unroll
                        for (int j=0; j<4; j++){
                            rowResult[j][c] = 1 - probs[c]; //match
                        }
                    }
                }


                //figure out where to save float4 in shared memory to allow coalesced read access to shared memory
                //read access should be coalesced within the whole warp, not only within the group

                constexpr int numAccesses = numRegs/4;

                const int accessChunk = i;
                const int accessChunkIdInThread = accessChunk % numAccesses;
                const int targetThreadIdInGroup = accessChunk / numAccesses;
                const int targetThreadIdInWarp = threadGroupIdInWarp * group_size + targetThreadIdInGroup;

                const int outputAccessChunk = accessChunkIdInThread * warpsize + targetThreadIdInWarp;
                const int outputCol = outputAccessChunk;

                // if(blockIdx.x == 0){
                //     printf("threadGroupId %d, i %d, targetThreadIdInGroup %d, targetThreadIdInWarp %d, outputAccessChunk %d\n", 
                //         threadGroupIdInWarp, i,targetThreadIdInGroup, targetThreadIdInWarp, outputAccessChunk );
                // }

                // if(threadIdInGroup == 0){
                //     printf("float kernel permuted grouped pssm\n");
                //     for(int r = 0; r < 5; r++){
                //         for(int c = 0; c < 16*numRegs; c++){
                //             printf("%f %f ", lambda_array[r][c].x, lambda_array[r][c].y);
                //         }
                //         printf("\n");
                //     }
                // }

                #pragma unroll
                for (int j=0; j<5; j++){
                    float4* rowPtr = (float4*)(&lambda_array_permuted[j]);
                    rowPtr[outputCol] = *((float4*)&rowResult[j][0]);
                }
            }

            __syncwarp(myGroupMask);
        };

        auto load_PSSM = [&](){
            construct_PSSM_warp_coalesced();
        };

        auto load_probabilities = [&]() {
            char4 temp0, temp1;
            const char4* InsQualsAsChar4 = reinterpret_cast<const char4*>(&ins_quals[byteOffsetForRead]);
            const char4* DelQualsAsChar4 = reinterpret_cast<const char4*>(&del_quals[byteOffsetForRead]);
            for (int i=0; i<numRegs/4; i++) {
                if (threadIdInGroup*numRegs/4+i < (readLength+3)/4) {

                    temp0 = InsQualsAsChar4[threadIdInGroup*numRegs/4+i];
                    temp1 = DelQualsAsChar4[threadIdInGroup*numRegs/4+i];

                    delta[4*i] = cPH2PR[uint8_t(temp0.x)];
                    delta[4*i+1] = cPH2PR[uint8_t(temp0.y)];
                    delta[4*i+2] = cPH2PR[uint8_t(temp0.z)];
                    delta[4*i+3] = cPH2PR[uint8_t(temp0.w)];
            //        delta[2*i] = __floats2half2_rn(cPH2PR[uint8_t(temp0.x)],cPH2PR[uint8_t(temp0.y)]);
            //        delta[2*i+1] = __floats2half2_rn(cPH2PR[uint8_t(temp0.z)],cPH2PR[uint8_t(temp0.w)]);

                    sigma[4*i] = cPH2PR[uint8_t(temp1.x)];
                    sigma[4*i+1] = cPH2PR[uint8_t(temp1.y)];
                    sigma[4*i+2] = cPH2PR[uint8_t(temp1.z)];
                    sigma[4*i+3] = cPH2PR[uint8_t(temp1.w)];
            //        sigma[2*i] = __floats2half2_rn(cPH2PR[uint8_t(temp1.x)],cPH2PR[uint8_t(temp1.y)]);
            //        sigma[2*i+1] = __floats2half2_rn(cPH2PR[uint8_t(temp1.z)],cPH2PR[uint8_t(temp1.w)]);

                    alpha[4*i] = 1.0 - (delta[4*i] + sigma[4*i]);
                    alpha[4*i+1] = 1.0 - (delta[4*i+1] + sigma[4*i+1]);
                    alpha[4*i+2] = 1.0 - (delta[4*i+2] + sigma[4*i+2]);
                    alpha[4*i+3] = 1.0 - (delta[4*i+3] + sigma[4*i+3]);
                //    alpha[2*i] = __float2half2_rn(1.0) - __hadd2(delta[2*i], sigma[2*i]);
                //    alpha[2*i+1] = __float2half2_rn(1.0) - __hadd2(delta[2*i+1], sigma[2*i+1]);
                }
            }

        };

        auto init_penalties = [&]() {
            #pragma unroll
            for (int i=0; i<numRegs; i++) M[i] = D[i] = Results[i] = 0.0;
            M_l = M_ul = D_ul = I_ul = D_l = I = 0.0;
            if (!threadIdInGroup) D_l = D_ul = init_D;
        };


        char hap_letter;

        
        auto calc_DP_float = [&](int row){

            // #define PRINT_COMPUTATION
            #ifdef PRINT_COMPUTATION
            float IArray[numRegs];
            #endif


            
            //warp coalesced
            float4* sbt_row = (float4*)(&lambda_array_permuted[hap_letter]);
            float4 lambda = *((float4*)(&sbt_row[0 * warpsize + threadIdInWarp]));

            // if(group_size == 8 && numRegs == 20){
            //     if(blockIdx.x == 0){
            //         printf("thread %d, load from %p, bank %lu\n", 
            //             threadIdInWarp, &sbt_row[0 * warpsize + threadIdInWarp], (size_t(&sbt_row[0 * warpsize + threadIdInWarp])/4) % 32);
            //     }
            // }
            
            //memcpy(&score2, &lambda.x, sizeof(float2));
            penalty_temp0 = M[0];
            penalty_temp1 = D[0];
            M[0] = lambda.x * fmaf(alpha[0],M_ul,beta*(I_ul+D_ul));
            D[0] = fmaf(sigma[0],penalty_temp0,eps*D[0]);
            I = fmaf(delta[0],M_ul,eps*I_ul);
            #ifdef PRINT_COMPUTATION
            IArray[0] = I;
            #endif
            Results[0] += M[0] + I;
            penalty_temp2 = M[1];
            penalty_temp3 = D[1];
            M[1] = lambda.y * fmaf(alpha[1],penalty_temp0,beta*(I+penalty_temp1));
            D[1] = fmaf(sigma[1],penalty_temp2,eps*D[1]);
            I = fmaf(delta[1],penalty_temp0,eps*I);
            #ifdef PRINT_COMPUTATION
            IArray[1] = I;
            #endif
            Results[1] += M[1] + I;

            //memcpy(&score2, &lambda.z, sizeof(float2));
            penalty_temp0 = M[2];
            penalty_temp1 = D[2];
            M[2] = lambda.z * fmaf(alpha[2],penalty_temp2,beta*(I+penalty_temp3));
            D[2] = fmaf(sigma[2],penalty_temp0,eps*D[2]);
            I = fmaf(delta[2],penalty_temp2,eps*I);
            #ifdef PRINT_COMPUTATION
            IArray[2] = I;
            #endif
            Results[2] += M[2] + I;
            penalty_temp2 = M[3];
            penalty_temp3 = D[3];
            M[3] = lambda.w * fmaf(alpha[3],penalty_temp0,beta*(I+penalty_temp1));
            D[3] = fmaf(sigma[3],penalty_temp2,eps*D[3]);
            I = fmaf(delta[3],penalty_temp0,eps*I);
            #ifdef PRINT_COMPUTATION
            IArray[3] = I;
            #endif
            Results[3] += M[3] + I;

            // if(threadIdInGroup * numRegs + 0 < haploLength) printf("row %d, col %d, lambda %f\n", row, threadIdInGroup * numRegs + 0, lambda.x);
            // if(threadIdInGroup * numRegs + 1 < haploLength) printf("row %d, col %d, lambda %f\n", row, threadIdInGroup * numRegs + 1, lambda.y);
            // if(threadIdInGroup * numRegs + 2 < haploLength) printf("row %d, col %d, lambda %f\n", row, threadIdInGroup * numRegs + 2, lambda.z);
            // if(threadIdInGroup * numRegs + 3 < haploLength) printf("row %d, col %d, lambda %f\n", row, threadIdInGroup * numRegs + 3, lambda.w);

            #pragma unroll
            for (int i=1; i<numRegs/4; i++) {
                float4 lambda = *((float4*)(&sbt_row[i * warpsize + threadIdInWarp]));

                // if(group_size == 8 && numRegs == 20){
                //     if(blockIdx.x == 0){
                //         printf("thread %d, load from %p, bank %lu\n", 
                //             threadIdInWarp, &sbt_row[i * warpsize + threadIdInWarp], (size_t(&sbt_row[i * warpsize + threadIdInWarp])/4) % 32);
                //     }
                // }
                

                //memcpy(&score2, &lambda.x, sizeof(float2));
                penalty_temp0 = M[4*i];
                penalty_temp1 = D[4*i];
                M[4*i] = lambda.x * fmaf(alpha[4*i],penalty_temp2,beta*(I+penalty_temp3));
                D[4*i] = fmaf(sigma[4*i],penalty_temp0,eps*D[4*i]);
                I = fmaf(delta[4*i],penalty_temp2,eps*I);
                #ifdef PRINT_COMPUTATION
                IArray[4*i] = I;
                #endif
                Results[4*i] += M[4*i] + I;
                penalty_temp2 = M[4*i+1];
                penalty_temp3 = D[4*i+1];
                M[4*i+1] = lambda.y * fmaf(alpha[4*i+1],penalty_temp0,beta*(I+penalty_temp1));
                D[4*i+1] = fmaf(sigma[4*i+1],penalty_temp2,eps*D[4*i+1]);
                I = fmaf(delta[4*i+1],penalty_temp0,eps*I);
                #ifdef PRINT_COMPUTATION
                IArray[4*i+1] = I;
                #endif
                Results[4*i+1] += M[4*i+1] + I;

                //memcpy(&score2, &lambda.z, sizeof(float2));
                penalty_temp0 = M[4*i+2];
                penalty_temp1 = D[4*i+2];
                M[4*i+2] = lambda.z * fmaf(alpha[4*i+2],penalty_temp2,beta*(I+penalty_temp3));
                D[4*i+2] = fmaf(sigma[4*i+2],penalty_temp0,eps*D[4*i+2]);
                I = fmaf(delta[4*i+2],penalty_temp2,eps*I);
                #ifdef PRINT_COMPUTATION
                IArray[4*i+2] = I;
                #endif
                Results[4*i+2] += M[4*i+2] + I;

                penalty_temp2 = M[4*i+3];
                penalty_temp3 = D[4*i+3];
                M[4*i+3] = lambda.w * fmaf(alpha[4*i+3],penalty_temp0,beta*(I+penalty_temp1));
                D[4*i+3] = fmaf(sigma[4*i+3],penalty_temp2,eps*D[4*i+3]);
                I = fmaf(delta[4*i+3],penalty_temp0,eps*I);
                #ifdef PRINT_COMPUTATION
                IArray[4*i+3] = I;
                #endif
                Results[4*i+3] += M[4*i+3] + I;

            //     if(threadIdInGroup * numRegs + 4*i + 0 < haploLength) printf("row %d, col %d, lambda %f\n", row, threadIdInGroup * numRegs + 4*i + 0, lambda.x);
            //     if(threadIdInGroup * numRegs + 4*i + 1 < haploLength) printf("row %d, col %d, lambda %f\n", row, threadIdInGroup * numRegs + 4*i + 1, lambda.y);
            //     if(threadIdInGroup * numRegs + 4*i + 2 < haploLength) printf("row %d, col %d, lambda %f\n", row, threadIdInGroup * numRegs + 4*i + 2, lambda.z);
            //     if(threadIdInGroup * numRegs + 4*i + 3 < haploLength) printf("row %d, col %d, lambda %f\n", row, threadIdInGroup * numRegs + 4*i + 3, lambda.w);
            }

            #ifdef PRINT_COMPUTATION
                // if(threadIdInGroup == 0){
                //     printf("M:\n");
                // };
                // __syncwarp(myGroupMask);
                // for(int t = group_size-1; t >= 0 ; t--){
                //     if(t == threadIdInGroup){
                //         for(int i = 0; i < t * numRegs; i++){
                //             printf("                                         ");
                //         }
                //         for(int i = 0; i < numRegs; i++){
                //             printf("%.3f ", M[i]);
                //         }
                //         printf("\n");
                //     }
                //     __syncwarp(myGroupMask);
                // }
                // if(threadIdInGroup == 0){
                //     printf("\n");
                // };
                // __syncwarp(myGroupMask);

                // if(threadIdInGroup == 0){
                //     printf("I:\n");
                // };
                // __syncwarp(myGroupMask);
                // for(int t = group_size-1; t >= 0 ; t--){
                //     if(t == threadIdInGroup){
                //         for(int i = 0; i < t * numRegs; i++){
                //             printf("                                         ");
                //         }
                //         for(int i = 0; i < numRegs; i++){
                //             printf("%.3f ", IArray[i]);
                //         }
                //         printf("\n");
                //     }
                //     __syncwarp(myGroupMask);
                // }
                // if(threadIdInGroup == 0){
                //     printf("\n");
                // };
                // __syncwarp(myGroupMask);


                if(threadIdInGroup == 0){
                    printf("M:\n");
                };
                __syncwarp(myGroupMask);
                //for(int t = 0; t <= group_size-1 ; t++){
                for(int t = 0; t <= result_thread ; t++){
                    if(t == threadIdInGroup){
                        printf("thread %d. ", threadIdInGroup);
                        for(int i = 0; i < numRegs; i++){
                            printf("%.3f ", M[i]);
                        }
                        printf("\n");
                    }
                    __syncwarp(myGroupMask);
                }
                if(threadIdInGroup == 0){
                    printf("\n");
                };
                __syncwarp(myGroupMask);

                if(threadIdInGroup == 0){
                    printf("I:\n");
                };
                __syncwarp(myGroupMask);
                //for(int t = 0; t <= group_size-1 ; t++){
                for(int t = 0; t <= result_thread ; t++){
                    if(t == threadIdInGroup){
                        printf("thread %d. ", threadIdInGroup);
                        for(int i = 0; i < numRegs; i++){
                            printf("%.3f ", IArray[i]);
                        }
                        printf("\n");
                    }
                    __syncwarp(myGroupMask);
                }
                if(threadIdInGroup == 0){
                    printf("\n");
                };
                __syncwarp(myGroupMask);


                if(threadIdInGroup == 0){
                    printf("D:\n");
                };
                __syncwarp(myGroupMask);
                //for(int t = 0; t <= group_size-1 ; t++){
                for(int t = 0; t <= result_thread ; t++){
                    if(t == threadIdInGroup){
                        printf("thread %d. ", threadIdInGroup);
                        for(int i = 0; i < numRegs; i++){
                            printf("%.3f ", D[i]);
                        }
                        printf("\n");
                    }
                    __syncwarp(myGroupMask);
                }
                if(threadIdInGroup == 0){
                    printf("\n");
                };
                __syncwarp(myGroupMask);


                
                if(threadIdInGroup == 0){
                    printf("Current results:\n");
                };
                __syncwarp(myGroupMask);
                for(int t = 0; t < group_size-1; t++){
                    if(t == threadIdInGroup){
                        for(int i = 0; i < numRegs; i++){
                            printf("%.6f ", Results[i]);
                        }
                    }
                    __syncwarp(myGroupMask);
                }
                if(threadIdInGroup == 0){
                    printf("\n");
                };
                __syncwarp(myGroupMask);
            #endif
        };

        auto shuffle_penalty = [&]() {
            M_ul = M_l;
            D_ul = D_l;

            M_l = __shfl_up_sync(myGroupMask, M[numRegs-1], 1, group_size);
            I_ul = __shfl_up_sync(myGroupMask, I, 1, group_size);
            D_l = __shfl_up_sync(myGroupMask, D[numRegs-1], 1, group_size);

            if (!threadIdInGroup) {
                M_l = I_ul = 0.0;
                D_l = init_D;
            }
        };

        

        load_PSSM();
        load_probabilities();
        // compute_probabilities();

        init_D = constant/haploLength;
        init_penalties();

        char4 new_hap_letter4;
        hap_letter = 4;
        int k;
        for (k=0; k<haploLength-3; k+=4) {
            new_hap_letter4 = HapsAsChar4[k/4];
            hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
            if (!threadIdInGroup) hap_letter = new_hap_letter4.x;
            calc_DP_float(k);
            shuffle_penalty();

            hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
            if (!threadIdInGroup) hap_letter = new_hap_letter4.y;
            calc_DP_float(k+1);
            shuffle_penalty();

            hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
            if (!threadIdInGroup) hap_letter = new_hap_letter4.z;
            calc_DP_float(k+2);
            shuffle_penalty();

            hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
            if (!threadIdInGroup) hap_letter = new_hap_letter4.w;
            calc_DP_float(k+3);
            shuffle_penalty();
        }
        if (haploLength%4 >= 1) {
            new_hap_letter4 = HapsAsChar4[k/4];
            hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
            if (!threadIdInGroup) hap_letter = new_hap_letter4.x;
            calc_DP_float(k);
            shuffle_penalty();
        }
        if (haploLength%4 >= 2) {
            hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
            if (!threadIdInGroup) hap_letter = new_hap_letter4.y;
            calc_DP_float(k+1);
            shuffle_penalty();
        }
        if (haploLength%4 >= 3) {
            hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
            if (!threadIdInGroup) hap_letter = new_hap_letter4.z;
            calc_DP_float(k+2);
            shuffle_penalty();
        }
        for (k=0; k<result_thread; k++) {
            // hap_letter = __shfl_up_sync(__activemask(), hap_letter, 1, 32);
            hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
            calc_DP_float(haploLength+k);
            shuffle_penalty(); // shuffle_penalty_active();
        }


        // adjust I values
        I = fmaf(delta[0],M_ul,eps*I_ul);
        #ifdef PRINT_COMPUTATION
        if(threadIdInGroup == result_thread){
            printf("final I compute: %f\n", I);
        }
        #endif
        Results[0] += I;
        I = fmaf(delta[1],M[0],eps*I);
        #ifdef PRINT_COMPUTATION
        if(threadIdInGroup == result_thread){
            printf("final I compute: %f\n", I);
        }
        #endif
        Results[1] += I;
        for (int p=1; p<numRegs/2; p++) {
            I = fmaf(delta[2*p],M[2*p-1],eps*I);
            #ifdef PRINT_COMPUTATION
            if(threadIdInGroup == result_thread){
                printf("final I compute: %f\n", I);
            }
            #endif
            Results[2*p] += I;
            I = fmaf(delta[2*p+1],M[2*p],eps*I);
            #ifdef PRINT_COMPUTATION
            if(threadIdInGroup == result_thread){
                printf("final I compute: %f\n", I);
            }
            #endif
            Results[2*p+1] += I;
        }

        //repurpose shared memory to stage output. 
        //since the output register index is computed at runtime, 
        //the compiler frequently stores the Results in local memory to be able to load the specific value at the end.
        //doing this once manually in shared memory avoids the atuomatic stores to local memory
        __syncwarp(myGroupMask);
        float* smemOutputBuffer = &lambda_array_permuted[0][0];


        if (threadIdInGroup == result_thread) {
            // float temp_res = Results[result_reg];

            float4* smemOutputBuffer4 = (float4*)smemOutputBuffer;
            #pragma unroll
            for(int i = 0; i < numRegs/4; i++){
                //need to ensure that we only access smem elements which are used by the group. here we use the same access pattern as during computations (warp striped)
                smemOutputBuffer4[i * warpsize + threadIdInWarp] = *((float4*)&Results[4*i]);
            }
            float temp_res = smemOutputBuffer[4*(result_reg/4 * warpsize + threadIdInWarp) + (result_reg % 4)];

            temp_res =  log10f(temp_res) - log10f(constant);
            resultoutput[resultOutputIndex] = temp_res;
        }
    }


}


template <int group_size, int numRegs> 
__global__
void PairHMM_align_partition_float_allowMultipleSequenceGroupsPerWarp_coalesced_smem_improvedResultComputation(
    float* resultoutput,
    const uint8_t* readData,
    const uint8_t* hapData,
    const uint8_t* base_quals,
    const uint8_t* ins_quals,
    const uint8_t* del_quals,
    const int* readBeginOffsets,
    const int* hapBeginOffsets,
    const int* readLengths,
    const int* hapLengths,
    const int* numHapsPerSequenceGroup,
    const int* numHapsPerSequenceGroupPrefixSum,
    const int* indicesPerSequenceGroup,
    const int* numReadsPerSequenceGroupPrefixSum,
    const int numSequenceGroups,
    const int* resultOffsetsPerSequenceGroup,
    const int* numAlignmentsPerSequenceGroup,
    const int* numAlignmentsPerSequenceGroupInclusivePrefixSum,
    const int numAlignments
) {
    static_assert(numRegs % 4 == 0);

    constexpr int warpsize = 32;
    constexpr int blocksize = 32;
    constexpr int numGroupsPerBlock = blocksize / group_size;

    constexpr int rowsize = numGroupsPerBlock * group_size*numRegs;
    alignas(16) __shared__ float lambda_array_permuted[5][rowsize];

    float M[numRegs], I, D[numRegs];
    float alpha[numRegs], delta[numRegs], sigma[numRegs];
    float epsArray[numRegs];

    const int threadIdInGroup = threadIdx.x % group_size;
    // const int threadGroupIdInBlock = threadIdx.x / group_size;
    const int threadIdInWarp = threadIdx.x % warpsize;
    const int threadGroupIdInWarp = threadIdInWarp / group_size;
    const int threadGroupIdInGrid = (threadIdx.x + blockIdx.x * blockDim.x) / group_size;
    const unsigned int myGroupMask = __match_any_sync(0xFFFFFFFF, threadGroupIdInGrid); //compute mask for all threads with same threadGroupIdInGrid
    
    // const int numGroupsInGrid = blockDim.x * gridDim.x / group_size;
    // for(int alignmentId = threadGroupIdInGrid; alignmentId < SDIV(numAlignments,numGroupsPerBlock)*numGroupsPerBlock; alignmentId += numGroupsInGrid){
    //     if(alignmentId >= numAlignments) continue;
        //if(alignmentId != 4) continue;
    const int alignmentId = threadGroupIdInGrid;
    if(alignmentId < numAlignments){

        const int sequenceGroupIdByThreadGroupId = thrust::distance(
            numAlignmentsPerSequenceGroupInclusivePrefixSum,
            thrust::upper_bound(thrust::seq,
                numAlignmentsPerSequenceGroupInclusivePrefixSum,
                numAlignmentsPerSequenceGroupInclusivePrefixSum + numSequenceGroups,
                alignmentId
            )
        );
        const int sequenceGroupId = min(sequenceGroupIdByThreadGroupId, numSequenceGroups-1);
        const int threadGroupIdInSequenceGroup = alignmentId - (sequenceGroupId == 0 ? 0 : numAlignmentsPerSequenceGroupInclusivePrefixSum[sequenceGroupId-1]);
        const int hapToProcessInSequenceGroup = threadGroupIdInSequenceGroup % numHapsPerSequenceGroup[sequenceGroupId];
        const int readIndexToProcessInSequenceGroup = threadGroupIdInSequenceGroup / numHapsPerSequenceGroup[sequenceGroupId];

        const int readIndexOffset = numReadsPerSequenceGroupPrefixSum[sequenceGroupId];
        const int readIndexOffset_inChunk = readIndexOffset - numReadsPerSequenceGroupPrefixSum[0];
        const int readToProcessInSequenceGroup = indicesPerSequenceGroup[readIndexOffset_inChunk + readIndexToProcessInSequenceGroup];

        const int read_nr = readToProcessInSequenceGroup;
        // const int global_read_id = read_nr + readIndexOffset;
        const int read_id_inChunk = read_nr + readIndexOffset_inChunk;


        const int byteOffsetForRead = readBeginOffsets[read_id_inChunk];
        const int readLength = readLengths[read_id_inChunk];

        const int b_h_off = numHapsPerSequenceGroupPrefixSum[sequenceGroupId];
        const int b_h_off_inChunk = b_h_off - numHapsPerSequenceGroupPrefixSum[0];
        const int bytesOffsetForHap = hapBeginOffsets[hapToProcessInSequenceGroup+b_h_off_inChunk];
        const char4* const HapsAsChar4 = reinterpret_cast<const char4*>(&hapData[bytesOffsetForHap]);
        const int haploLength = hapLengths[hapToProcessInSequenceGroup+b_h_off_inChunk];

        const int resultOutputIndex = resultOffsetsPerSequenceGroup[sequenceGroupId] + read_nr*numHapsPerSequenceGroup[sequenceGroupId]+hapToProcessInSequenceGroup;

        const int threadColumnOffset = threadIdInGroup * numRegs;

        // if(false && threadIdx.x == 7 && partitionId == 4 && alignmentId == 4){
        //     printf("new kernel\n");

        //     printf("cPH2PR\n");
        //     for(int i = 0; i < 128; i++){
        //         printf("%f ", (cPH2PR[i]));
        //     }
        //     printf("\n");

        //     printf("read\n");
        //     for(int i = 0; i < readLength; i++){
        //         printf("%d", int(readData[byteOffsetForRead + i]));
        //     }
        //     printf("\n");
        //     printf("base\n");
        //     for(int i = 0; i < readLength; i++){
        //         printf("%c", (base_quals[byteOffsetForRead + i])+33);
        //     }
        //     printf("\n");
        //     printf("ins\n");
        //     for(int i = 0; i < readLength; i++){
        //         printf("%c", (ins_quals[byteOffsetForRead + i])+33);
        //     }
        //     printf("\n");
        //     printf("del\n");
        //     for(int i = 0; i < readLength; i++){
        //         printf("%c", (del_quals[byteOffsetForRead + i])+33);
        //     }
        //     printf("\n");
        //     printf("hap\n");
        //     for(int i = 0; i < haploLength; i++){
        //         printf("%d", int(hapData[bytesOffsetForHap + i]));
        //     }
        //     printf("\n");
        // }


        const float eps = 0.1;
        const float beta = 0.9;
        float M_l, D_l, M_ul, D_ul, I_ul;
        float penalty_temp0, penalty_temp1, penalty_temp2, penalty_temp3;
        float init_D;

        float myResult = 0.0f;

        const float constant = ::cuda::std::numeric_limits<float>::max() / 16;
        // const float constant = 1;

        const int result_thread = (readLength-1)/numRegs;
        const int result_reg = (readLength-1)%numRegs;

        auto construct_PSSM_warp_coalesced = [&](){
            __syncwarp(myGroupMask);
            
            const char4* QualsAsChar4 = reinterpret_cast<const char4*>(&base_quals[byteOffsetForRead]);
            const char4* ReadsAsChar4 = reinterpret_cast<const char4*>(&readData[byteOffsetForRead]);
            for (int i=threadIdInGroup; i < group_size * numRegs / 4; i+=group_size) {
                alignas(16) float rowResult[5][4];
                #pragma unroll 
                for(int j = 0; j < 5; j++){
                    rowResult[j][0] = 0.f;
                    rowResult[j][1] = 0.f;
                    rowResult[j][2] = 0.f;
                    rowResult[j][3] = 0.f;
                    rowResult[j][4] = 0.f;
                }
                if(i < (readLength+3)/4){
                    const char4 temp0 = QualsAsChar4[i];
                    const char4 temp1 = ReadsAsChar4[i];
                    alignas(4) char quals[4];
                    memcpy(&quals[0], &temp0, sizeof(char4));
                    alignas(4) char letters[4];
                    memcpy(&letters[0], &temp1, sizeof(char4));

                    float probs[4];
                    #pragma unroll
                    for(int c = 0; c < 4; c++){
                        probs[c] = cPH2PR[quals[c]];
                        //if(isnan(probs[c])){printf("nan line %d\n", __LINE__); }
                    }

                    #pragma unroll
                    for(int c = 0; c < 4; c++){
                        //hap == N always matches
                        rowResult[4][c] = 1 - probs[c]; //match
                        //if(isnan(rowResult[4][c])){printf("nan line %d\n", __LINE__); }

                        if(letters[c] < 4){
                            // set hap == read to 1 - prob, hap != read to prob / 3
                            #pragma unroll
                            for (int j=0; j<4; j++){
                                rowResult[j][c] = (j == letters[c]) ? 1 - probs[c] : probs[c]/3.0f; //match or mismatch
                                //if(isnan(rowResult[j][c])){printf("nan line %d\n", __LINE__); }
                            }
                        }else{
                            // read == N always matches
                            #pragma unroll
                            for (int j=0; j<4; j++){
                                rowResult[j][c] = 1 - probs[c]; //match
                                //if(isnan(rowResult[j][c])){printf("nan line %d\n", __LINE__); }
                            }
                        }
                    }
                }

                //figure out where to save float4 in shared memory to allow coalesced read access to shared memory
                //read access should be coalesced within the whole warp, not only within the group

                constexpr int numAccesses = numRegs/4;

                const int accessChunk = i;
                const int accessChunkIdInThread = accessChunk % numAccesses;
                const int targetThreadIdInGroup = accessChunk / numAccesses;
                const int targetThreadIdInWarp = threadGroupIdInWarp * group_size + targetThreadIdInGroup;

                const int outputAccessChunk = accessChunkIdInThread * warpsize + targetThreadIdInWarp;
                const int outputCol = outputAccessChunk;

                // if(blockIdx.x == 0){
                //     printf("threadGroupId %d, i %d, targetThreadIdInGroup %d, targetThreadIdInWarp %d, outputAccessChunk %d\n", 
                //         threadGroupIdInWarp, i,targetThreadIdInGroup, targetThreadIdInWarp, outputAccessChunk );
                // }

                // if(threadIdInGroup == 0){
                //     printf("float kernel permuted grouped pssm\n");
                //     for(int r = 0; r < 5; r++){
                //         for(int c = 0; c < 16*numRegs; c++){
                //             printf("%f %f ", lambda_array[r][c].x, lambda_array[r][c].y);
                //         }
                //         printf("\n");
                //     }
                // }

                #pragma unroll
                for (int j=0; j<5; j++){
                    float4* rowPtr = (float4*)(&lambda_array_permuted[j]);
                    rowPtr[outputCol] = *((float4*)&rowResult[j][0]);
                }
            }

            __syncwarp(myGroupMask);
        };

        auto load_PSSM = [&](){
            construct_PSSM_warp_coalesced();
        };

        auto load_probabilities = [&]() {
            /*
                Initialize alpha, sigma, delta, and epsilon

                oob cells for delta and epsilon are initialized such that for each row
                the value which needs to be added to the final result (i.e. I and M of last column)
                is propagated to the last per-thread column register.
            */


            char4 temp0{};
            char4 temp1{};
            const char4* InsQualsAsChar4 = reinterpret_cast<const char4*>(&ins_quals[byteOffsetForRead]);
            const char4* DelQualsAsChar4 = reinterpret_cast<const char4*>(&del_quals[byteOffsetForRead]);
            for (int i=0; i<numRegs/4; i++) {
                if (threadIdInGroup*numRegs/4+i < (readLength+3)/4) {
                    temp0 = InsQualsAsChar4[threadIdInGroup*numRegs/4+i];
                    temp1 = DelQualsAsChar4[threadIdInGroup*numRegs/4+i];
                }

                //for the first oob column, set delta to 1, for other oob columns set delta to 0
                delta[4*i] = (threadColumnOffset + 4*i < readLength) ? cPH2PR[uint8_t(temp0.x)] : ((threadColumnOffset + 4*i == readLength) ? 1.f : 0.f);
                delta[4*i+1] = (threadColumnOffset + 4*i+1 < readLength) ? cPH2PR[uint8_t(temp0.y)] : ((threadColumnOffset + 4*i+1 == readLength) ? 1.f : 0.f);
                delta[4*i+2] = (threadColumnOffset + 4*i+2 < readLength) ? cPH2PR[uint8_t(temp0.z)] : ((threadColumnOffset + 4*i+2 == readLength) ? 1.f : 0.f);
                delta[4*i+3] = (threadColumnOffset + 4*i+3 < readLength) ? cPH2PR[uint8_t(temp0.w)] : ((threadColumnOffset + 4*i+3 == readLength) ? 1.f : 0.f);

                sigma[4*i] = cPH2PR[uint8_t(temp1.x)];
                sigma[4*i+1] = cPH2PR[uint8_t(temp1.y)];
                sigma[4*i+2] = cPH2PR[uint8_t(temp1.z)];
                sigma[4*i+3] = cPH2PR[uint8_t(temp1.w)];

                // if(isnan(delta[4*i])){printf("nan line %d\n", __LINE__); }
                // if(isnan(delta[4*i+1])){printf("nan line %d\n", __LINE__); }
                // if(isnan(delta[4*i+2])){printf("nan line %d\n", __LINE__); }
                // if(isnan(delta[4*i+3])){printf("nan line %d\n", __LINE__); }
                // if(isnan(sigma[4*i])){printf("nan line %d\n", __LINE__); }
                // if(isnan(sigma[4*i+1])){printf("nan line %d\n", __LINE__); }
                // if(isnan(sigma[4*i+2])){printf("nan line %d\n", __LINE__); }
                // if(isnan(sigma[4*i+3])){printf("nan line %d\n", __LINE__); }
            }
            for (int i=0; i<numRegs/4; i++) {
                alpha[4*i] = 1.0f - (delta[4*i] + sigma[4*i]);
                alpha[4*i+1] = 1.0f - (delta[4*i+1] + sigma[4*i+1]);
                alpha[4*i+2] = 1.0f - (delta[4*i+2] + sigma[4*i+2]);
                alpha[4*i+3] = 1.0f - (delta[4*i+3] + sigma[4*i+3]);

                //set epsilon to 1 for all oob columns
                epsArray[4*i] = (threadColumnOffset + 4*i < readLength) ? eps : 1.0f;
                epsArray[4*i+1] = (threadColumnOffset + 4*i+1 < readLength) ? eps : 1.0f;
                epsArray[4*i+2] = (threadColumnOffset + 4*i+2 < readLength) ? eps : 1.0f;
                epsArray[4*i+3] = (threadColumnOffset + 4*i+3 < readLength) ? eps : 1.0f;

                // if(isnan(alpha[4*i])){printf("nan line %d\n", __LINE__); }
                // if(isnan(alpha[4*i+1])){printf("nan line %d\n", __LINE__); }
                // if(isnan(alpha[4*i+2])){printf("nan line %d\n", __LINE__); }
                // if(isnan(alpha[4*i+3])){printf("nan line %d\n", __LINE__); }
                // if(isnan(epsArray[4*i])){printf("nan line %d\n", __LINE__); }
                // if(isnan(epsArray[4*i+1])){printf("nan line %d\n", __LINE__); }
                // if(isnan(epsArray[4*i+2])){printf("nan line %d\n", __LINE__); }
                // if(isnan(epsArray[4*i+3])){printf("nan line %d\n", __LINE__); }
            }

            // if(threadIdInGroup == 0){
            //     printf("delta\n");
            // }
            // __syncwarp(myGroupMask);
            // for(int t = 0; t <= result_thread; t++){
            //     if(t == threadIdInGroup){
            //         for(int i = 0; i < numRegs; i++){
            //             printf("%f ", delta[i]);
            //         }
            //     }
            //     __syncwarp(myGroupMask);
            // }
            // if(threadIdInGroup == 0){
            //     printf("\n");
            // }
            // if(threadIdInGroup == 0){
            //     printf("epsArray\n");
            // }
            // __syncwarp(myGroupMask);
            // for(int t = 0; t <= result_thread; t++){
            //     if(t == threadIdInGroup){
            //         for(int i = 0; i < numRegs; i++){
            //             printf("%f ", epsArray[i]);
            //         }
            //     }
            //     __syncwarp(myGroupMask);
            // }
            // if(threadIdInGroup == 0){
            //     printf("\n");
            // }
        };

        auto init_penalties = [&]() {
            #pragma unroll
            for (int i=0; i<numRegs; i++){
                M[i] = 0.0f;
            }
            for (int i=0; i<numRegs; i++){
                D[i] = 0.0f;
            }

            M_l = M_ul = D_ul = I_ul = D_l = I = 0.0;
            if (!threadIdInGroup) D_l = D_ul = init_D;
        };


        char hap_letter;

        // #define PRINT_COMPUTATION
        
        auto calc_DP_float = [&](int row){
            #ifdef PRINT_COMPUTATION
            if(threadIdInGroup == 0){
                printf("tileNr %d, row %d\n", 0, row);
            }
            #endif

            /*
                Perform the relaxation for the current diagonal
                Note: I is computed for the PREVIOUS diagonal. This ensures that initialiation values remain valid 
                for threads which are still out-of-bounds during the first groupsize-1 diagonals
            */

            #ifdef PRINT_COMPUTATION
            float IArray[numRegs];
            #endif
            
            //warp coalesced
            float4* sbt_row = (float4*)(&lambda_array_permuted[hap_letter]);
            float4 lambda = *((float4*)(&sbt_row[0 * warpsize + threadIdInWarp]));

            // if(threadIdx.x == 7 && partitionId == 4 && alignmentId == 4 && row < 20){
            //     printf("row %d, load index %d, lambda %f %f %f %f\n", row, 0 * warpsize + threadIdInWarp, lambda.x, lambda.y, lambda.z, lambda.w);
            // }
            
            penalty_temp0 = M[0];
            penalty_temp1 = D[0];
            M[0] = lambda.x * fmaf(alpha[0],M_ul,beta*(I_ul+D_ul));
            D[0] = fmaf(sigma[0],penalty_temp0,epsArray[0]*D[0]);
            I = fmaf(delta[0],M_ul,epsArray[0]*I_ul); // As explained above, we use the left neighbors M and I of the previous row
            #ifdef PRINT_COMPUTATION
            IArray[0] = I;
            #endif

            penalty_temp2 = M[1];
            penalty_temp3 = D[1];
            M[1] = lambda.y * fmaf(alpha[1],penalty_temp0,beta*(I+penalty_temp1));
            D[1] = fmaf(sigma[1],penalty_temp2,epsArray[1]*D[1]);
            I = fmaf(delta[1],penalty_temp0,epsArray[1]*I);
            #ifdef PRINT_COMPUTATION
            IArray[1] = I;
            #endif

            penalty_temp0 = M[2];
            penalty_temp1 = D[2];
            M[2] = lambda.z * fmaf(alpha[2],penalty_temp2,beta*(I+penalty_temp3));
            D[2] = fmaf(sigma[2],penalty_temp0,epsArray[2]*D[2]);
            I = fmaf(delta[2],penalty_temp2,epsArray[2]*I);
            #ifdef PRINT_COMPUTATION
            IArray[2] = I;
            #endif

            penalty_temp2 = M[3];
            penalty_temp3 = D[3];
            M[3] = lambda.w * fmaf(alpha[3],penalty_temp0,beta*(I+penalty_temp1));
            D[3] = fmaf(sigma[3],penalty_temp2,epsArray[3]*D[3]);
            I = fmaf(delta[3],penalty_temp0,epsArray[3]*I);
            #ifdef PRINT_COMPUTATION
            IArray[3] = I;
            #endif

            #pragma unroll
            for (int i=1; i<numRegs/4; i++) {
                float4 lambda = *((float4*)(&sbt_row[i * warpsize + threadIdInWarp]));
                // if(threadIdx.x == 7 && partitionId == 4 && alignmentId == 4 && row < 20){
                //     printf("row %d, load index %d, lambda %f %f %f %f\n", row, i * warpsize + threadIdInWarp, lambda.x, lambda.y, lambda.z, lambda.w);
                // }

                penalty_temp0 = M[4*i];
                penalty_temp1 = D[4*i];
                M[4*i] = lambda.x * fmaf(alpha[4*i],penalty_temp2,beta*(I+penalty_temp3));
                D[4*i] = fmaf(sigma[4*i],penalty_temp0,epsArray[4*i]*D[4*i]);
                I = fmaf(delta[4*i],penalty_temp2,epsArray[4*i]*I);
                #ifdef PRINT_COMPUTATION
                IArray[4*i] = I;
                #endif

                penalty_temp2 = M[4*i+1];
                penalty_temp3 = D[4*i+1];
                M[4*i+1] = lambda.y * fmaf(alpha[4*i+1],penalty_temp0,beta*(I+penalty_temp1));
                D[4*i+1] = fmaf(sigma[4*i+1],penalty_temp2,epsArray[4*i+1]*D[4*i+1]);
                I = fmaf(delta[4*i+1],penalty_temp0,epsArray[4*i+1]*I);
                #ifdef PRINT_COMPUTATION
                IArray[4*i+1] = I;
                #endif

                penalty_temp0 = M[4*i+2];
                penalty_temp1 = D[4*i+2];
                M[4*i+2] = lambda.z * fmaf(alpha[4*i+2],penalty_temp2,beta*(I+penalty_temp3));
                D[4*i+2] = fmaf(sigma[4*i+2],penalty_temp0,epsArray[4*i+2]*D[4*i+2]);
                I = fmaf(delta[4*i+2],penalty_temp2,epsArray[4*i+2]*I);
                #ifdef PRINT_COMPUTATION
                IArray[4*i+2] = I;
                #endif

                penalty_temp2 = M[4*i+3];
                penalty_temp3 = D[4*i+3];
                M[4*i+3] = lambda.w * fmaf(alpha[4*i+3],penalty_temp0,beta*(I+penalty_temp1));
                // if(threadIdx.x == 7 && partitionId == 4 && alignmentId == 4 && row >= 209 && row <= 210){
                //     printf("line %d. %f = %f * fma(%f, %f, %f*(%f+%f))\n", __LINE__, M[4*i+3], lambda.w, alpha[4*i+3], penalty_temp0, beta, I, penalty_temp1);
                //     printf("line %d. fma(%f, %f, %f*%f)\n", __LINE__, sigma[4*i+3], penalty_temp2, epsArray[4*i+3], D[4*i+3]);
                // }
                D[4*i+3] = fmaf(sigma[4*i+3],penalty_temp2,epsArray[4*i+3]*D[4*i+3]);
                I = fmaf(delta[4*i+3],penalty_temp0,epsArray[4*i+3]*I);
                #ifdef PRINT_COMPUTATION
                IArray[4*i+3] = I;
                #endif

            }
            // if(threadIdx.x == 7 && partitionId == 4 && alignmentId == 4 && row >= 209 && row <= 210){
            //     printf("row %d\n", row);
            //     printf("new M:\n");
            //     for(int i = 0; i < numRegs; i++){
            //         printf("%f ", M[i]);
            //     }
            //     printf("\n");
            //     printf("new D:\n");
            //     for(int i = 0; i < numRegs; i++){
            //         printf("%f ", D[i]);
            //     }
            //     printf("\n");
            // }

            myResult += I;
            if(result_reg == numRegs-1){
                myResult += M[numRegs-1];
            }

            #ifdef PRINT_COMPUTATION
                // if(threadIdInGroup == 0){
                //     printf("M:\n");
                // };
                // __syncwarp(myGroupMask);
                // for(int t = group_size-1; t >= 0 ; t--){
                //     if(t == threadIdInGroup){
                //         for(int i = 0; i < t * numRegs; i++){
                //             printf("                                         ");
                //         }
                //         for(int i = 0; i < numRegs; i++){
                //             printf("%.3f ", M[i]);
                //         }
                //         printf("\n");
                //     }
                //     __syncwarp(myGroupMask);
                // }
                // if(threadIdInGroup == 0){
                //     printf("\n");
                // };
                // __syncwarp(myGroupMask);

                // if(threadIdInGroup == 0){
                //     printf("I:\n");
                // };
                // __syncwarp(myGroupMask);
                // for(int t = group_size-1; t >= 0 ; t--){
                //     if(t == threadIdInGroup){
                //         for(int i = 0; i < t * numRegs; i++){
                //             printf("                                         ");
                //         }
                //         for(int i = 0; i < numRegs; i++){
                //             printf("%.3f ", IArray[i]);
                //         }
                //         printf("\n");
                //     }
                //     __syncwarp(myGroupMask);
                // }
                // if(threadIdInGroup == 0){
                //     printf("\n");
                // };
                // __syncwarp(myGroupMask);

                if(threadIdInGroup == 0){
                    printf("M:\n");
                };
                __syncwarp(myGroupMask);
                //for(int t = 0; t <= group_size-1 ; t++){
                for(int t = 0; t <= result_thread ; t++){
                    if(t == threadIdInGroup){
                        printf("thread %d. ", threadIdInGroup);
                        for(int i = 0; i < numRegs; i++){
                            printf("%40.40f ", M[i]);
                        }
                        printf("\n");
                    }
                    __syncwarp(myGroupMask);
                }
                if(threadIdInGroup == 0){
                    printf("\n");
                };
                __syncwarp(myGroupMask);

                if(threadIdInGroup == 0){
                    printf("I:\n");
                };
                __syncwarp(myGroupMask);
                //for(int t = 0; t <= group_size-1 ; t++){
                for(int t = 0; t <= result_thread ; t++){
                    if(t == threadIdInGroup){
                        printf("thread %d. ", threadIdInGroup);
                        for(int i = 0; i < numRegs; i++){
                            printf("%40.40f ", IArray[i]);
                        }
                        printf("\n");
                    }
                    __syncwarp(myGroupMask);
                }
                if(threadIdInGroup == 0){
                    printf("\n");
                };
                __syncwarp(myGroupMask);

                if(threadIdInGroup == 0){
                    printf("D:\n");
                };
                __syncwarp(myGroupMask);
                //for(int t = 0; t <= group_size-1 ; t++){
                for(int t = 0; t <= result_thread ; t++){
                    if(t == threadIdInGroup){
                        printf("thread %d. ", threadIdInGroup);
                        for(int i = 0; i < numRegs; i++){
                            printf("%40.40f ", D[i]);
                        }
                        printf("\n");
                    }
                    __syncwarp(myGroupMask);
                }
                if(threadIdInGroup == 0){
                    printf("\n");
                };
                __syncwarp(myGroupMask);

                printf("thread %d, myResult %40.40f\n", threadIdInGroup, myResult);
            #endif

        };

        auto shuffle_penalty = [&]() {
            M_ul = M_l;
            D_ul = D_l;

            M_l = __shfl_up_sync(myGroupMask, M[numRegs-1], 1, group_size);
            I_ul = __shfl_up_sync(myGroupMask, I, 1, group_size);
            D_l = __shfl_up_sync(myGroupMask, D[numRegs-1], 1, group_size);

            if (!threadIdInGroup) {
                M_l = I_ul = 0.0;
                D_l = init_D;
            }
        };



        load_PSSM();
        load_probabilities();
        // compute_probabilities();

        init_D = constant/haploLength;
        init_penalties();

        char4 new_hap_letter4;
        hap_letter = 4;
        int k;
        for (k=0; k<haploLength-3; k+=4) {
            new_hap_letter4 = HapsAsChar4[k/4];
            hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
            if (!threadIdInGroup) hap_letter = new_hap_letter4.x;
            calc_DP_float(k);
            shuffle_penalty();

            hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
            if (!threadIdInGroup) hap_letter = new_hap_letter4.y;
            calc_DP_float(k+1);
            shuffle_penalty();

            hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
            if (!threadIdInGroup) hap_letter = new_hap_letter4.z;
            calc_DP_float(k+2);
            shuffle_penalty();

            hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
            if (!threadIdInGroup) hap_letter = new_hap_letter4.w;
            calc_DP_float(k+3);
            shuffle_penalty();
        }
        if (haploLength%4 >= 1) {
            new_hap_letter4 = HapsAsChar4[k/4];
            hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
            if (!threadIdInGroup) hap_letter = new_hap_letter4.x;
            calc_DP_float(k);
            shuffle_penalty();
        }
        if (haploLength%4 >= 2) {
            hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
            if (!threadIdInGroup) hap_letter = new_hap_letter4.y;
            calc_DP_float(k+1);
            shuffle_penalty();
        }
        if (haploLength%4 >= 3) {
            hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
            if (!threadIdInGroup) hap_letter = new_hap_letter4.z;
            calc_DP_float(k+2);
            shuffle_penalty();
        }
        for (k=0; k<result_thread; k++) {
            hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
            calc_DP_float(haploLength+k);
            shuffle_penalty(); 
        }

        // Relaxation computes I for the previous diagonal. Thus, we are missing I for the last diagonal.
        // Compute it now!

        I = fmaf(delta[0],M_ul,epsArray[0]*I_ul);
        #ifdef PRINT_COMPUTATION
        if(threadIdInGroup == result_thread){
            printf("final I compute: %40.40f\n", I);
        }
        #endif
        I = fmaf(delta[1],M[0],epsArray[1]*I);
        #ifdef PRINT_COMPUTATION
        if(threadIdInGroup == result_thread){
            printf("final I compute: %40.40f\n", I);
        }
        #endif
        for (int p=1; p<numRegs/2; p++) {
            I = fmaf(delta[2*p],M[2*p-1],epsArray[2*p]*I);
            #ifdef PRINT_COMPUTATION
            if(threadIdInGroup == result_thread){
                printf("final I compute: %40.40f\n", I);
            }
            #endif
            I = fmaf(delta[2*p+1],M[2*p],epsArray[2*p+1]*I);
            #ifdef PRINT_COMPUTATION
            if(threadIdInGroup == result_thread){
                printf("final I compute: %40.40f\n", I);
            }
            #endif
        }

        myResult += I;
        #undef PRINT_COMPUTATION

        // bool error = false;
        if (threadIdInGroup == result_thread) {
            float temp_res = myResult;
            temp_res =  log10f(temp_res) - log10f(constant);
            resultoutput[resultOutputIndex] = temp_res;
            // if(isnan(temp_res)){
            //     int foo = atomicAdd(&dcounter, 1);
            //     if(foo == 0){
            //         printf("alignmentId %d\n", alignmentId);
            //         printf("cPH2PR\n");
            //         for(int i = 0; i < 128; i++){
            //             printf("%f ", (cPH2PR[i]));
            //         }
            //         printf("\n");

            //         printf("read\n");
            //         for(int i = 0; i < readLength; i++){
            //             printf("%d", int(readData[byteOffsetForRead + i]));
            //         }
            //         printf("\n");
            //         printf("base\n");
            //         for(int i = 0; i < readLength; i++){
            //             printf("%c", (base_quals[byteOffsetForRead + i])+33);
            //         }
            //         printf("\n");
            //         printf("ins\n");
            //         for(int i = 0; i < readLength; i++){
            //             printf("%c", (ins_quals[byteOffsetForRead + i])+33);
            //         }
            //         printf("\n");
            //         printf("del\n");
            //         for(int i = 0; i < readLength; i++){
            //             printf("%c", (del_quals[byteOffsetForRead + i])+33);
            //         }
            //         printf("\n");
            //         printf("hap\n");
            //         for(int i = 0; i < haploLength; i++){
            //             printf("%d", int(hapData[bytesOffsetForHap + i]));
            //         }
            //         printf("\n");
            //         printf("nan\n");
            //     }
            //     error = true;
            // }
        }
    }


}

template <int group_size, int numRegs> 
constexpr auto PairHMM_float_kernel
    = PairHMM_align_partition_float_allowMultipleSequenceGroupsPerWarp_coalesced_smem_improvedResultComputation<group_size, numRegs>;
    // = PairHMM_align_partition_float_allowMultipleSequenceGroupsPerWarp_coalesced_smem_noimprovedResultComputation<group_size, numRegs>;




//perform groupwide reduction using Op. Return result in all threads of group
template<class T, class Op>
__device__
T group_reduce_broadcast(unsigned int groupmask, T value, int groupsize, Op op){
    for (int i=1; i<groupsize; i*=2){
        value = op(value, __shfl_xor_sync(groupmask, value, i, groupsize));
    }
    return value;
}


template<int numRegs>
struct Coefficients{
    float beta;
    float alpha[numRegs];
    float delta[numRegs];
    float sigma[numRegs];
    float epsArray[numRegs];
};

template<int group_size, int numRegs, class LambdaArray>
struct PairHMMState{
    float M[numRegs], I, D[numRegs];
    float M_ul;
    float M_l;
    float I_ul;
    float I_l;
    float D_ul;
    float D_l;
    float myResult = 0;

    int result_thread;
    unsigned int myGroupMask;
    // LambdaArray lambda_array_permuted;
    // Coefficients<numRegs> coeffs;
    
    const LambdaArray& lambda_array_permuted;
    const Coefficients<numRegs>& coeffs;

    __device__
    PairHMMState(const LambdaArray& lambda_array_permuted_, const Coefficients<numRegs>& coeffs_, int result_thread_, unsigned int myGroupMask_)
        : result_thread(result_thread_), myGroupMask(myGroupMask_), lambda_array_permuted(lambda_array_permuted_), coeffs(coeffs_)
    {}

    // #define PRINT_COMPUTATION

    __device__ __forceinline__
    void relax_I_row_behind(int row, int tileNr, int hap_letter, int threadIdInWarp, int threadIdInGroup, int result_reg, bool isLastTile){
        constexpr int warpsize = 32;
        #ifdef PRINT_COMPUTATION
        if(threadIdInGroup == 0){
            printf("tileNr %d, row %d\n", tileNr, row);
        }
        #endif
        const auto& alpha = coeffs.alpha;
        const auto& beta = coeffs.beta;
        const auto& delta = coeffs.delta;
        const auto& epsArray = coeffs.epsArray;
        const auto& sigma = coeffs.sigma;
        float penalty_temp0, penalty_temp1, penalty_temp2, penalty_temp3;

        /*
            Perform the relaxation for the current diagonal
            Note: I is computed for the PREVIOUS diagonal. This ensures that initialiation values remain valid 
            for threads which are still out-of-bounds during the first groupsize-1 diagonals

            The register suffixes _ul and _l indicate upper-left neighbor and left neighbor, respectively, 
            from the view of the first M value to be computes.
            Because of this diagonal shift, this means that I_ul is the upper-left neighbor of M, but the left neighbor of I
        */

        #ifdef PRINT_COMPUTATION
        float IArray[numRegs];
        #endif
        
        //warp coalesced
        float4* sbt_row = (float4*)(&lambda_array_permuted[hap_letter][0]);
        float4 lambda = *((float4*)(&sbt_row[0 * warpsize + threadIdInWarp]));
        
        penalty_temp0 = M[0];
        penalty_temp1 = D[0];
        M[0] = lambda.x * fmaf(alpha[0],M_ul,beta*(I_ul+D_ul));
        // printf("M[0] = %f * fmaf(%f, %f, %f * (%f+%f))\n", lambda.x, alpha[0], M_ul, beta, I_ul, D_ul);
        D[0] = fmaf(sigma[0],penalty_temp0,epsArray[0]*D[0]);
        I = fmaf(delta[0],M_ul,epsArray[0]*I_ul); // As explained above, we use the left neighbors M and I of the previous row
        #ifdef PRINT_COMPUTATION
        IArray[0] = I;
        #endif

        penalty_temp2 = M[1];
        penalty_temp3 = D[1];
        M[1] = lambda.y * fmaf(alpha[1],penalty_temp0,beta*(I+penalty_temp1));
        D[1] = fmaf(sigma[1],penalty_temp2,epsArray[1]*D[1]);
        I = fmaf(delta[1],penalty_temp0,epsArray[1]*I);
        #ifdef PRINT_COMPUTATION
        IArray[1] = I;
        #endif

        penalty_temp0 = M[2];
        penalty_temp1 = D[2];
        M[2] = lambda.z * fmaf(alpha[2],penalty_temp2,beta*(I+penalty_temp3));
        D[2] = fmaf(sigma[2],penalty_temp0,epsArray[2]*D[2]);
        I = fmaf(delta[2],penalty_temp2,epsArray[2]*I);
        #ifdef PRINT_COMPUTATION
        IArray[2] = I;
        #endif

        penalty_temp2 = M[3];
        penalty_temp3 = D[3];
        M[3] = lambda.w * fmaf(alpha[3],penalty_temp0,beta*(I+penalty_temp1));
        D[3] = fmaf(sigma[3],penalty_temp2,epsArray[3]*D[3]);
        I = fmaf(delta[3],penalty_temp0,epsArray[3]*I);
        #ifdef PRINT_COMPUTATION
        IArray[3] = I;
        #endif

        #pragma unroll
        for (int i=1; i<numRegs/4; i++) {
            float4 lambda = *((float4*)(&sbt_row[i * warpsize + threadIdInWarp]));

            penalty_temp0 = M[4*i];
            penalty_temp1 = D[4*i];
            M[4*i] = lambda.x * fmaf(alpha[4*i],penalty_temp2,beta*(I+penalty_temp3));
            D[4*i] = fmaf(sigma[4*i],penalty_temp0,epsArray[4*i]*D[4*i]);
            I = fmaf(delta[4*i],penalty_temp2,epsArray[4*i]*I);
            #ifdef PRINT_COMPUTATION
            IArray[4*i] = I;
            #endif

            penalty_temp2 = M[4*i+1];
            penalty_temp3 = D[4*i+1];
            M[4*i+1] = lambda.y * fmaf(alpha[4*i+1],penalty_temp0,beta*(I+penalty_temp1));
            D[4*i+1] = fmaf(sigma[4*i+1],penalty_temp2,epsArray[4*i+1]*D[4*i+1]);
            I = fmaf(delta[4*i+1],penalty_temp0,epsArray[4*i+1]*I);
            #ifdef PRINT_COMPUTATION
            IArray[4*i+1] = I;
            #endif

            penalty_temp0 = M[4*i+2];
            penalty_temp1 = D[4*i+2];
            M[4*i+2] = lambda.z * fmaf(alpha[4*i+2],penalty_temp2,beta*(I+penalty_temp3));
            D[4*i+2] = fmaf(sigma[4*i+2],penalty_temp0,epsArray[4*i+2]*D[4*i+2]);
            I = fmaf(delta[4*i+2],penalty_temp2,epsArray[4*i+2]*I);
            #ifdef PRINT_COMPUTATION
            IArray[4*i+2] = I;
            #endif

            penalty_temp2 = M[4*i+3];
            penalty_temp3 = D[4*i+3];
            M[4*i+3] = lambda.w * fmaf(alpha[4*i+3],penalty_temp0,beta*(I+penalty_temp1));
            D[4*i+3] = fmaf(sigma[4*i+3],penalty_temp2,epsArray[4*i+3]*D[4*i+3]);
            I = fmaf(delta[4*i+3],penalty_temp0,epsArray[4*i+3]*I);
            #ifdef PRINT_COMPUTATION
            IArray[4*i+3] = I;
            #endif

        }

        if(isLastTile){
            myResult += I;
            if(result_reg == numRegs-1){
                myResult += M[numRegs-1];
            }
        }

        #ifdef PRINT_COMPUTATION
            // if(threadIdInGroup == 0){
            //     printf("M:\n");
            // };
            // __syncwarp(myGroupMask);
            // for(int t = group_size-1; t >= 0 ; t--){
            //     if(t == threadIdInGroup){
            //         for(int i = 0; i < t * numRegs; i++){
            //             printf("                                         ");
            //         }
            //         for(int i = 0; i < numRegs; i++){
            //             printf("%.3f ", M[i]);
            //         }
            //         printf("\n");
            //     }
            //     __syncwarp(myGroupMask);
            // }
            // if(threadIdInGroup == 0){
            //     printf("\n");
            // };
            // __syncwarp(myGroupMask);

            // if(threadIdInGroup == 0){
            //     printf("I:\n");
            // };
            // __syncwarp(myGroupMask);
            // for(int t = group_size-1; t >= 0 ; t--){
            //     if(t == threadIdInGroup){
            //         for(int i = 0; i < t * numRegs; i++){
            //             printf("                                         ");
            //         }
            //         for(int i = 0; i < numRegs; i++){
            //             printf("%.3f ", IArray[i]);
            //         }
            //         printf("\n");
            //     }
            //     __syncwarp(myGroupMask);
            // }
            // if(threadIdInGroup == 0){
            //     printf("\n");
            // };
            // __syncwarp(myGroupMask);

            if(threadIdInGroup == 0){
                printf("M:\n");
            };
            __syncwarp(myGroupMask);
            for(int t = 0; t <= group_size-1 ; t++){
            // for(int t = 0; t <= result_thread ; t++){
                if(t == threadIdInGroup){
                    printf("thread %d. ", threadIdInGroup);
                    for(int i = 0; i < numRegs; i++){
                        printf("%40.40f ", M[i]);
                    }
                    printf("\n");
                }
                __syncwarp(myGroupMask);
            }
            if(threadIdInGroup == 0){
                printf("\n");
            };
            __syncwarp(myGroupMask);

            if(threadIdInGroup == 0){
                printf("I:\n");
            };
            __syncwarp(myGroupMask);
            for(int t = 0; t <= group_size-1 ; t++){
            // for(int t = 0; t <= result_thread ; t++){
                if(t == threadIdInGroup){
                    printf("thread %d. ", threadIdInGroup);
                    for(int i = 0; i < numRegs; i++){
                        printf("%40.40f ", IArray[i]);
                    }
                    printf("\n");
                }
                __syncwarp(myGroupMask);
            }
            if(threadIdInGroup == 0){
                printf("\n");
            };
            __syncwarp(myGroupMask);

            if(threadIdInGroup == 0){
                printf("D:\n");
            };
            __syncwarp(myGroupMask);
            for(int t = 0; t <= group_size-1 ; t++){
            // for(int t = 0; t <= result_thread ; t++){
                if(t == threadIdInGroup){
                    printf("thread %d. ", threadIdInGroup);
                    for(int i = 0; i < numRegs; i++){
                        printf("%40.40f ", D[i]);
                    }
                    printf("\n");
                }
                __syncwarp(myGroupMask);
            }
            if(threadIdInGroup == 0){
                printf("\n");
            };
            __syncwarp(myGroupMask);

            printf("thread %d, myResult %40.40f\n", threadIdInGroup, myResult);
        #endif
    }

    __device__ __forceinline__
    void relax_final_I_row(int row, int tileNr, int hap_letter, int threadIdInWarp, int threadIdInGroup, int result_reg, bool isLastTile){
        #ifdef PRINT_COMPUTATION
        if(threadIdInGroup == 0){
            printf("relax_final_I_row tile %d, row %d\n", tileNr, row);
        }
        #endif
        #ifdef PRINT_COMPUTATION
        float IArray[numRegs];
        #endif

        const auto& delta = coeffs.delta;
        const auto& epsArray = coeffs.epsArray;
        I = fmaf(delta[0],M_ul,epsArray[0]*I_ul);
        #ifdef PRINT_COMPUTATION
        IArray[0] = I;
        #endif
        I = fmaf(delta[1],M[0],epsArray[1]*I);
        #ifdef PRINT_COMPUTATION
        IArray[1] = I;
        #endif
        for (int p=1; p<numRegs/2; p++) {
            I = fmaf(delta[2*p],M[2*p-1],epsArray[2*p]*I);
            #ifdef PRINT_COMPUTATION
            IArray[2*p] = I;
            #endif
            I = fmaf(delta[2*p+1],M[2*p],epsArray[2*p+1]*I);
            #ifdef PRINT_COMPUTATION
            IArray[2*p+1] = I;
            #endif
        }

        if(isLastTile){
            myResult += I;
        }

        #ifdef PRINT_COMPUTATION
            if(threadIdInGroup == 0){
                printf("final I:\n");
            };
            __syncwarp(myGroupMask);
            for(int t = 0; t <= group_size-1 ; t++){
            // for(int t = 0; t <= result_thread ; t++){
                if(t == threadIdInGroup){
                    printf("thread %d. ", threadIdInGroup);
                    for(int i = 0; i < numRegs; i++){
                        printf("%40.40f ", IArray[i]);
                    }
                    printf("\n");
                }
                __syncwarp(myGroupMask);
            }
            if(threadIdInGroup == 0){
                printf("\n");
            };
            __syncwarp(myGroupMask);

            printf("thread %d, myResult %40.40f\n", threadIdInGroup, myResult);
        #endif
    }

    #undef PRINT_COMPUTATION
};


template<int numGroupsPerBlock, int group_size, int numRegs>
struct LambdaArray{
    static constexpr int rowsize = numGroupsPerBlock * group_size * numRegs;
    alignas(16) float data[5][rowsize];

    __host__ __device__
    float* operator[](int r){ return &data[r][0]; }
    __host__ __device__
    const float* operator[](int r) const { return &data[r][0]; }
};


//uses a grid-strided loop. Can use less groups than alignments to limit temporary storage
template <int group_size, int numRegs> 
__global__
void PairHMM_float_multitile_kernel(
    float* resultoutput,
    const uint8_t* readData,
    const uint8_t* hapData,
    const uint8_t* base_quals,
    const uint8_t* ins_quals,
    const uint8_t* del_quals,
    const int* readBeginOffsets,
    const int* hapBeginOffsets,
    const int* readLengths,
    const int* hapLengths,
    const int* numHapsPerSequenceGroup,
    const int* numHapsPerSequenceGroupPrefixSum,
    const int* indicesPerSequenceGroup,
    const int* numReadsPerSequenceGroupPrefixSum,
    const int numSequenceGroups,
    const int* resultOffsetsPerSequenceGroup,
    const int* numAlignmentsPerSequenceGroup,
    const int* numAlignmentsPerSequenceGroupInclusivePrefixSum,
    const int numAlignments,
    char* globalTempStorage, //size tempBytesPerGroup * num groups in grid
    size_t tempBytesPerGroup, //sizeof(float) * 3 * (longest haplo length + 1 + group_size)
    int longestHaploLength
) {
    static_assert(numRegs % 4 == 0);

    constexpr int warpsize = 32;
    constexpr int blocksize = 32;
    constexpr int numGroupsPerBlock = blocksize / group_size;
    constexpr int tileSize = group_size * numRegs;

    // constexpr int rowsize = numGroupsPerBlock * group_size*numRegs;
    // alignas(16) __shared__ float lambda_array_permuted[5][rowsize];

    using LambdaArray = LambdaArray<numGroupsPerBlock, group_size, numRegs>;
    __shared__ LambdaArray lambda_array_permuted;

    
    const int threadIdInGroup = threadIdx.x % group_size;
    // const int threadGroupIdInBlock = threadIdx.x / group_size;
    const int threadIdInWarp = threadIdx.x % warpsize;
    const int threadGroupIdInWarp = threadIdInWarp / group_size;
    const int threadGroupIdInGrid = (threadIdx.x + blockIdx.x * blockDim.x) / group_size;
    const unsigned int myGroupMask = __match_any_sync(0xFFFFFFFF, threadGroupIdInGrid); //compute mask for all threads with same threadGroupIdInGrid
    const int numGroupsInGrid = blockDim.x * gridDim.x / group_size;
    
    const size_t tempStorageRows = longestHaploLength + 1 + group_size;
    char* const globalTempStorage_MD = globalTempStorage;
    char* const globalTempStorage_I = globalTempStorage_MD + sizeof(float2) * tempStorageRows * numGroupsInGrid;
    float2* const groupGlobalTempStorage_MD = reinterpret_cast<float2*>(globalTempStorage_MD) + tempStorageRows * threadGroupIdInGrid;
    float* const groupGlobalTempStorage_I = reinterpret_cast<float*>(globalTempStorage_I) + tempStorageRows * threadGroupIdInGrid;

    
    for(int alignmentId = threadGroupIdInGrid; alignmentId < numAlignments; alignmentId += numGroupsInGrid){
        
        const int sequenceGroupIdByThreadGroupId = thrust::distance(
            numAlignmentsPerSequenceGroupInclusivePrefixSum,
            thrust::upper_bound(thrust::seq,
                numAlignmentsPerSequenceGroupInclusivePrefixSum,
                numAlignmentsPerSequenceGroupInclusivePrefixSum + numSequenceGroups,
                alignmentId
            )
        );
        const int sequenceGroupId = min(sequenceGroupIdByThreadGroupId, numSequenceGroups-1);
        const int threadGroupIdInSequenceGroup = alignmentId - (sequenceGroupId == 0 ? 0 : numAlignmentsPerSequenceGroupInclusivePrefixSum[sequenceGroupId-1]);
        const int hapToProcessInSequenceGroup = threadGroupIdInSequenceGroup % numHapsPerSequenceGroup[sequenceGroupId];
        const int readIndexToProcessInSequenceGroup = threadGroupIdInSequenceGroup / numHapsPerSequenceGroup[sequenceGroupId];
        
        const int readIndexOffset = numReadsPerSequenceGroupPrefixSum[sequenceGroupId];
        const int readIndexOffset_inChunk = readIndexOffset - numReadsPerSequenceGroupPrefixSum[0];
        const int readToProcessInSequenceGroup = indicesPerSequenceGroup[readIndexOffset_inChunk + readIndexToProcessInSequenceGroup];
        
        const int read_nr = readToProcessInSequenceGroup;
        // const int global_read_id = read_nr + readIndexOffset;
        const int read_id_inChunk = read_nr + readIndexOffset_inChunk;
        const int byteOffsetForRead = readBeginOffsets[read_id_inChunk];
        const int readLength = readLengths[read_id_inChunk];
        
        const int b_h_off = numHapsPerSequenceGroupPrefixSum[sequenceGroupId];
        const int b_h_off_inChunk = b_h_off - numHapsPerSequenceGroupPrefixSum[0];
        const int bytesOffsetForHap = hapBeginOffsets[hapToProcessInSequenceGroup+b_h_off_inChunk];
        const char4* const HapsAsChar4 = reinterpret_cast<const char4*>(&hapData[bytesOffsetForHap]);
        const int haploLength = hapLengths[hapToProcessInSequenceGroup+b_h_off_inChunk];
        
        const int resultOutputIndex = resultOffsetsPerSequenceGroup[sequenceGroupId] + read_nr*numHapsPerSequenceGroup[sequenceGroupId]+hapToProcessInSequenceGroup;
        
        // if(threadIdx.x == 0){
        //     printf("read\n");
        //     for(int i = 0; i < readLength; i++){
        //         printf("%d ", int(readData[byteOffsetForRead + i]));
        //     }
        //     printf("\n");
        //     printf("hap\n");
        //     for(int i = 0; i < haploLength; i++){
        //         printf("%d ", int(hapData[bytesOffsetForHap + i]));
        //     }
        //     printf("\n");
        // }
        
        // const int threadColumnOffset = threadIdInGroup * numRegs;
        
        const int numTiles = SDIV(readLength, tileSize);
        const int maxNumTilesInWarp = group_reduce_broadcast(myGroupMask, numTiles, group_size, [](int a, int b){ return max(a,b); });
        
        float tileLastColumn_M = 0;
        float tileLastColumn_I = 0;
        float tileLastColumn_D = 0;
        float tileLeftBorder_M = 0;
        float tileLeftBorder_I = 0;
        float tileLeftBorder_D = 0;
        int tempWriteOffset = threadIdInGroup;
        int tempLoadOffset = threadIdInGroup;
        
        Coefficients<numRegs> coeffs;
        const float eps = 0.1;
        coeffs.beta = 0.9;

        
        const float constant = ::cuda::std::numeric_limits<float>::max() / 16;
        // const float constant = 1;
        const float init_D = constant/haploLength;
        
        
        const int readLengthInLastTile = readLength - (numTiles-1) * (tileSize);
        const int result_thread = (readLengthInLastTile-1)/numRegs;
        const int result_reg = (readLengthInLastTile-1)%numRegs;

        PairHMMState<group_size, numRegs, LambdaArray> state(lambda_array_permuted, coeffs, result_thread, myGroupMask);

        // #define PRINT_TEMPSTORAGE_ACCESS

        auto cacheTileLastColumnM = [&](){
            tileLastColumn_M = __shfl_down_sync(myGroupMask, tileLastColumn_M, 1, group_size);
            if(threadIdInGroup == group_size-1){
                tileLastColumn_M = state.M[numRegs-1];
            }
        };

        auto cacheTileLastColumnI = [&](){
            tileLastColumn_I = __shfl_down_sync(myGroupMask, tileLastColumn_I, 1, group_size);
            if(threadIdInGroup == group_size-1){
                tileLastColumn_I = state.I;
            }
        };

        auto cacheTileLastColumnD = [&](){
            tileLastColumn_D = __shfl_down_sync(myGroupMask, tileLastColumn_D, 1, group_size);
            if(threadIdInGroup == group_size-1){
                tileLastColumn_D = state.D[numRegs-1];
            }
        };

        auto cacheTileLastColumn = [&](){
            cacheTileLastColumnM();
            cacheTileLastColumnI();
            cacheTileLastColumnD();
        };

        auto writeCachedTileLastColumnToMemory = [&](){
            groupGlobalTempStorage_MD[tempWriteOffset] = make_float2(tileLastColumn_M, tileLastColumn_D);
            groupGlobalTempStorage_I[tempWriteOffset] = tileLastColumn_I;
            #ifdef PRINT_TEMPSTORAGE_ACCESS
            printf("write %40.40f %40.40f %40.40f to [%d]\n", tileLastColumn_M, tileLastColumn_I, tileLastColumn_D, tempWriteOffset);
            #endif
            tempWriteOffset += group_size;
        };

        auto writeCachedTileLastColumnToMemoryFinal = [&](int firstValidThread){
            groupGlobalTempStorage_MD[tempWriteOffset-firstValidThread] = make_float2(tileLastColumn_M, tileLastColumn_D);
            groupGlobalTempStorage_I[tempWriteOffset-firstValidThread] = tileLastColumn_I;
            #ifdef PRINT_TEMPSTORAGE_ACCESS
            printf("write %40.40f %40.40f %40.40f to [%d]\n", tileLastColumn_M, tileLastColumn_I, tileLastColumn_D, tempWriteOffset-firstValidThread);
            #endif
        };

        auto loadTileLeftBorderFromMemory = [&](){
            const float2 MD = groupGlobalTempStorage_MD[tempLoadOffset];
            tileLeftBorder_M = MD.x;
            tileLeftBorder_D = MD.y;
            tileLeftBorder_I = groupGlobalTempStorage_I[tempLoadOffset];
            #ifdef PRINT_TEMPSTORAGE_ACCESS
            printf("load %40.40f %40.40f %40.40f from [%d]\n", tileLeftBorder_M, tileLeftBorder_I, tileLeftBorder_D, tempLoadOffset);
            #endif
            tempLoadOffset += group_size;
        };

        auto leftBorderShuffleDown = [&](){
            tileLeftBorder_M = __shfl_down_sync(myGroupMask, tileLeftBorder_M, 1, group_size);
            tileLeftBorder_I = __shfl_down_sync(myGroupMask, tileLeftBorder_I, 1, group_size);
            tileLeftBorder_D = __shfl_down_sync(myGroupMask, tileLeftBorder_D, 1, group_size);
        };

        auto clearGlobalTempStorage = [&](int hapLength){
            if(threadIdInGroup < group_size - 1){
                groupGlobalTempStorage_MD[hapLength + threadIdInGroup] = float2{};
                groupGlobalTempStorage_I[hapLength + threadIdInGroup] = float{};
            }
        };

        auto construct_PSSM_warp_coalesced = [&](int tileNr){
            __syncwarp(myGroupMask);

            const int tileOffset = tileNr * group_size * numRegs;
            const int nextTileOffset = (tileNr+1) * group_size * numRegs;
            // const int threadColumnOffsetInTile = threadIdInGroup * numRegs;
            // const int threadColumnOffset = tileOffset + threadColumnOffsetInTile;
            
            const char4* QualsAsChar4 = reinterpret_cast<const char4*>(&base_quals[byteOffsetForRead]);
            const char4* ReadsAsChar4 = reinterpret_cast<const char4*>(&readData[byteOffsetForRead]);
            //for (int i=threadIdInGroup; i<(readLength+3)/4; i+=group_size) {
                // const char4 temp0 = QualsAsChar4[i];
                // const char4 temp1 = ReadsAsChar4[i];
            //for(int i = tileOffset + 4*threadIdInGroup; i < min(readLength, nextTileOffset); i += 4*group_size){
            for(int i = tileOffset + 4*threadIdInGroup; i < nextTileOffset; i += 4*group_size){
                alignas(16) float rowResult[5][4];
                #pragma unroll 
                for(int j = 0; j < 5; j++){
                    rowResult[j][0] = 0.f;
                    rowResult[j][1] = 0.f;
                    rowResult[j][2] = 0.f;
                    rowResult[j][3] = 0.f;
                    rowResult[j][4] = 0.f;
                }
                if(i < min(readLength, nextTileOffset)){

                    const char4 temp0 = QualsAsChar4[i/4];
                    const char4 temp1 = ReadsAsChar4[i/4];
                    alignas(4) char quals[4];
                    memcpy(&quals[0], &temp0, sizeof(char4));
                    alignas(4) char letters[4];
                    memcpy(&letters[0], &temp1, sizeof(char4));

                    float probs[4];
                    #pragma unroll
                    for(int c = 0; c < 4; c++){
                        probs[c] = cPH2PR[quals[c]];
                    }

                    #pragma unroll
                    for(int c = 0; c < 4; c++){
                        //hap == N always matches
                        rowResult[4][c] = 1 - probs[c]; //match

                        if(letters[c] < 4){
                            // set hap == read to 1 - prob, hap != read to prob / 3
                            #pragma unroll
                            for (int j=0; j<4; j++){
                                rowResult[j][c] = (j == letters[c]) ? 1 - probs[c] : probs[c]/3.0f; //match or mismatch
                            }
                        }else{
                            // read == N always matches
                            #pragma unroll
                            for (int j=0; j<4; j++){
                                rowResult[j][c] = 1 - probs[c]; //match
                            }
                        }
                    }
                }

                //figure out where to save float4 in shared memory to allow coalesced read access to shared memory
                //read access should be coalesced within the whole warp, not only within the group

                constexpr int numAccesses = numRegs/4;

                const int accessChunkInTile = (i-tileOffset)/4;
                const int accessChunkIdInThread = accessChunkInTile % numAccesses;
                const int targetThreadIdInGroup = accessChunkInTile / numAccesses;
                const int targetThreadIdInWarp = threadGroupIdInWarp * group_size + targetThreadIdInGroup;

                const int outputAccessChunk = accessChunkIdInThread * warpsize + targetThreadIdInWarp;
                const int outputCol = outputAccessChunk;

                // if(blockIdx.x == 0){
                //     printf("threadGroupId %d, i %d, targetThreadIdInGroup %d, targetThreadIdInWarp %d, outputAccessChunk %d\n", 
                //         threadGroupIdInWarp, i,targetThreadIdInGroup, targetThreadIdInWarp, outputAccessChunk );
                // }

                // if(threadIdInGroup == 0){
                //     printf("float kernel permuted grouped pssm\n");
                //     for(int r = 0; r < 5; r++){
                //         for(int c = 0; c < 16*numRegs; c++){
                //             printf("%f %f ", lambda_array[r][c].x, lambda_array[r][c].y);
                //         }
                //         printf("\n");
                //     }
                // }

                #pragma unroll
                for (int j=0; j<5; j++){
                    float4* rowPtr = (float4*)(&lambda_array_permuted[j][0]);
                    rowPtr[outputCol] = *((float4*)&rowResult[j][0]);
                }
            }

            __syncwarp(myGroupMask);
        };

        auto load_PSSM = [&](int tileNr){
            construct_PSSM_warp_coalesced(tileNr);
        };

        auto load_probabilities = [&](int tileNr) {
            auto& alpha = coeffs.alpha;
            auto& delta = coeffs.delta;
            auto& epsArray = coeffs.epsArray;
            auto& sigma = coeffs.sigma;

            /*
                Initialize alpha, sigma, delta, and epsilon

                oob cells for delta and epsilon are initialized such that for each row
                the value which needs to be added to the final result (i.e. I and M of last column)
                is propagated to the last per-thread column register.
            */

            const int tileOffset = tileNr * group_size * numRegs;
            const int threadColumnOffsetInTile = threadIdInGroup * numRegs;
            const int threadColumnOffset = tileOffset + threadColumnOffsetInTile;

            char4 temp0{};
            char4 temp1{};
            const char4* InsQualsAsChar4 = reinterpret_cast<const char4*>(&ins_quals[byteOffsetForRead]);
            const char4* DelQualsAsChar4 = reinterpret_cast<const char4*>(&del_quals[byteOffsetForRead]);
            for (int i=0; i<numRegs/4; i++) {
                if (threadColumnOffset + 4*i < readLength) {
                    temp0 = InsQualsAsChar4[threadColumnOffset/4+i];
                    temp1 = DelQualsAsChar4[threadColumnOffset/4+i];
                }

                //for the first oob column, set delta to 1, for other oob columns set delta to 0
                delta[4*i] = (threadColumnOffset + 4*i < readLength) ? cPH2PR[uint8_t(temp0.x)] : ((threadColumnOffset + 4*i == readLength) ? 1.f : 0.f);
                delta[4*i+1] = (threadColumnOffset + 4*i+1 < readLength) ? cPH2PR[uint8_t(temp0.y)] : ((threadColumnOffset + 4*i+1 == readLength) ? 1.f : 0.f);
                delta[4*i+2] = (threadColumnOffset + 4*i+2 < readLength) ? cPH2PR[uint8_t(temp0.z)] : ((threadColumnOffset + 4*i+2 == readLength) ? 1.f : 0.f);
                delta[4*i+3] = (threadColumnOffset + 4*i+3 < readLength) ? cPH2PR[uint8_t(temp0.w)] : ((threadColumnOffset + 4*i+3 == readLength) ? 1.f : 0.f);

                sigma[4*i] = cPH2PR[uint8_t(temp1.x)];
                sigma[4*i+1] = cPH2PR[uint8_t(temp1.y)];
                sigma[4*i+2] = cPH2PR[uint8_t(temp1.z)];
                sigma[4*i+3] = cPH2PR[uint8_t(temp1.w)];
            }
            for (int i=0; i<numRegs/4; i++) {
                alpha[4*i] = 1.0f - (delta[4*i] + sigma[4*i]);
                alpha[4*i+1] = 1.0f - (delta[4*i+1] + sigma[4*i+1]);
                alpha[4*i+2] = 1.0f - (delta[4*i+2] + sigma[4*i+2]);
                alpha[4*i+3] = 1.0f - (delta[4*i+3] + sigma[4*i+3]);

                //set epsilon to 1 for all oob columns
                epsArray[4*i] = (threadColumnOffset + 4*i < readLength) ? eps : 1.0f;
                epsArray[4*i+1] = (threadColumnOffset + 4*i+1 < readLength) ? eps : 1.0f;
                epsArray[4*i+2] = (threadColumnOffset + 4*i+2 < readLength) ? eps : 1.0f;
                epsArray[4*i+3] = (threadColumnOffset + 4*i+3 < readLength) ? eps : 1.0f;
            }

            // if(threadIdInGroup == 0){
            //     printf("delta\n");
            // }
            // __syncwarp(myGroupMask);
            // for(int t = 0; t <= group_size-1; t++){
            // //for(int t = 0; t <= result_thread; t++){
            //     if(t == threadIdInGroup){
            //         for(int i = 0; i < numRegs; i++){
            //             printf("%f ", delta[i]);
            //         }
            //     }
            //     __syncwarp(myGroupMask);
            // }
            // if(threadIdInGroup == 0){
            //     printf("\n");
            // }
            // if(threadIdInGroup == 0){
            //     printf("epsArray\n");
            // }
            // __syncwarp(myGroupMask);
            // for(int t = 0; t <= group_size-1; t++){
            // // for(int t = 0; t <= result_thread; t++){
            //     if(t == threadIdInGroup){
            //         for(int i = 0; i < numRegs; i++){
            //             printf("%f ", epsArray[i]);
            //         }
            //     }
            //     __syncwarp(myGroupMask);
            // }
            // if(threadIdInGroup == 0){
            //     printf("\n");
            // }
        };

        auto init_penalties_firstTile = [&]() {
            #pragma unroll
            for (int i=0; i<numRegs; i++){
                state.M[i] = 0.0f;
            }
            for (int i=0; i<numRegs; i++){
                state.D[i] = 0.0f;
            }

            state.M_ul = state.D_ul = state.I = 0.0f;
            state.M_l = 0.0f;
            state.I_ul = 0.0f;
            state.D_l = 0.0f;
            if (threadIdInGroup == 0){
                state.D_l = init_D;
                state.D_ul = init_D;
            }
        };

        auto updateStateFromLeftBorder = [&](float leftM, float upperleftI, float leftD){
            state.M_l = leftM;
            state.I_ul = upperleftI;
            state.D_l = leftD;
        };

        auto init_penalties_notFirstTile = [&](int /*tileNr*/, float leftM, float upperleftI, float leftD) {
            #pragma unroll
            for (int i=0; i<numRegs; i++){
                state.M[i] = 0.0f;
            }
            for (int i=0; i<numRegs; i++){
                state.D[i] = 0.0f;
            }

            state.M_ul = state.D_ul =  state.I = 0.0f;
            state.M_l = 0.0f;
            state.I_ul = 0.0f;
            state.D_l = 0.0f;
            if (threadIdInGroup == 0){
                updateStateFromLeftBorder(leftM, upperleftI, leftD);
            }
        };


        char hap_letter;

        auto shuffle_penalty_I_row_behind = [&](int tileNr) {
            auto& M_ul = state.M_ul;
            auto& M_l = state.M_l;
            auto& I_ul = state.I_ul;
            auto& D_ul = state.D_ul;
            auto& D_l = state.D_l;
            auto& I = state.I;
            auto& M = state.M;
            auto& D = state.D;

            M_ul = M_l;
            D_ul = D_l;

            M_l = __shfl_up_sync(myGroupMask, M[numRegs-1], 1, group_size);
            I_ul = __shfl_up_sync(myGroupMask, I, 1, group_size);
            D_l = __shfl_up_sync(myGroupMask, D[numRegs-1], 1, group_size);

            if (threadIdInGroup == 0) {
                M_l = 0.0f;
                I_ul = 0.0;
                D_l = init_D;
            }
        };


        auto shuffle_penalty_I_row_behind_notFirstTile = [&](int tileNr, float leftM, float upperleftI, float leftD) {
            auto& M_ul = state.M_ul;
            auto& M_l = state.M_l;
            auto& I_ul = state.I_ul;
            auto& D_ul = state.D_ul;
            auto& D_l = state.D_l;
            auto& I = state.I;
            auto& M = state.M;
            auto& D = state.D;

            M_ul = M_l;
            D_ul = D_l;

            M_l = __shfl_up_sync(myGroupMask, M[numRegs-1], 1, group_size);
            I_ul = __shfl_up_sync(myGroupMask, I, 1, group_size);
            D_l = __shfl_up_sync(myGroupMask, D[numRegs-1], 1, group_size);

            if (threadIdInGroup == 0){
                updateStateFromLeftBorder(leftM, upperleftI, leftD);
            }
        };

        if(maxNumTilesInWarp == 1){
            //process the single tile

            constexpr int tileNr = 0;
            constexpr bool isLastTile = true;

            load_PSSM(tileNr);
            load_probabilities(tileNr);
            
            init_penalties_firstTile();

            char4 new_hap_letter4;
            hap_letter = 4;
            int k;
            for (k=0; k<haploLength-3; k+=4) {
                new_hap_letter4 = HapsAsChar4[k/4];
                hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
                if (!threadIdInGroup) hap_letter = new_hap_letter4.x;
                state.relax_I_row_behind(k, tileNr, hap_letter, threadIdInWarp, threadIdInGroup, result_reg, isLastTile);
                shuffle_penalty_I_row_behind(tileNr);

                hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
                if (!threadIdInGroup) hap_letter = new_hap_letter4.y;
                state.relax_I_row_behind(k+1, tileNr, hap_letter, threadIdInWarp, threadIdInGroup, result_reg, isLastTile);
                shuffle_penalty_I_row_behind(tileNr);

                hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
                if (!threadIdInGroup) hap_letter = new_hap_letter4.z;
                state.relax_I_row_behind(k+2, tileNr, hap_letter, threadIdInWarp, threadIdInGroup, result_reg, isLastTile);
                shuffle_penalty_I_row_behind(tileNr);

                hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
                if (!threadIdInGroup) hap_letter = new_hap_letter4.w;
                state.relax_I_row_behind(k+3, tileNr, hap_letter, threadIdInWarp, threadIdInGroup, result_reg, isLastTile);
                shuffle_penalty_I_row_behind(tileNr);
            }
            if (haploLength%4 >= 1) {
                new_hap_letter4 = HapsAsChar4[k/4];
                hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
                if (!threadIdInGroup) hap_letter = new_hap_letter4.x;
                state.relax_I_row_behind(k, tileNr, hap_letter, threadIdInWarp, threadIdInGroup, result_reg, isLastTile);
                shuffle_penalty_I_row_behind(tileNr);
            }
            if (haploLength%4 >= 2) {
                hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
                if (!threadIdInGroup) hap_letter = new_hap_letter4.y;
                state.relax_I_row_behind(k+1, tileNr, hap_letter, threadIdInWarp, threadIdInGroup, result_reg, isLastTile);
                shuffle_penalty_I_row_behind(tileNr);
            }
            if (haploLength%4 >= 3) {
                hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
                if (!threadIdInGroup) hap_letter = new_hap_letter4.z;
                state.relax_I_row_behind(k+2, tileNr, hap_letter, threadIdInWarp, threadIdInGroup, result_reg, isLastTile);
                shuffle_penalty_I_row_behind(tileNr);
            }
            for (k=0; k<result_thread; k++) {
                hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
                state.relax_I_row_behind(haploLength+k, tileNr, hap_letter, threadIdInWarp, threadIdInGroup, result_reg, isLastTile);
                shuffle_penalty_I_row_behind(tileNr);
            }

            // Relaxation computes I for the previous diagonal. Thus, we are missing I for the last diagonal.
            // Compute it now!

            state.relax_final_I_row(haploLength + result_thread - 1, tileNr, hap_letter, threadIdInWarp, threadIdInGroup, result_reg, isLastTile);

            if (threadIdInGroup == result_thread) {
                float temp_res = state.myResult;
                temp_res =  log10f(temp_res) - log10f(constant);
                resultoutput[resultOutputIndex] = temp_res;
            }
        }else{
            //process tile 0
            {
                
                constexpr int tileNr = 0;
                constexpr bool isLastTile = false;
                // constexpr bool coversRow0Yes = true;
                // constexpr bool coversRow0No = false;
                tempWriteOffset = threadIdInGroup;

                load_PSSM(tileNr);
                load_probabilities(tileNr);
                
                init_penalties_firstTile();

                const int numRows = haploLength + (group_size-1) + 1;
                const int numHaploChar4s = SDIV(haploLength, 4);
                char4 new_hap_letter4;
                hap_letter = 4;
                int r = 1;

                // process first (group_size-1) diagonals for which some threads are out-of-bounds or in the pre-initialized row
                new_hap_letter4 = HapsAsChar4[(r-1)/4];
                hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
                if (!threadIdInGroup) hap_letter = new_hap_letter4.x;
                state.relax_I_row_behind(r, tileNr, hap_letter, threadIdInWarp, threadIdInGroup, result_reg, isLastTile);
                shuffle_penalty_I_row_behind(tileNr);
                r++;

                hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
                if (!threadIdInGroup) hap_letter = new_hap_letter4.y;
                state.relax_I_row_behind(r, tileNr, hap_letter, threadIdInWarp, threadIdInGroup, result_reg, isLastTile);
                shuffle_penalty_I_row_behind(tileNr);
                r++;

                hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
                if (!threadIdInGroup) hap_letter = new_hap_letter4.z;
                state.relax_I_row_behind(r, tileNr, hap_letter, threadIdInWarp, threadIdInGroup, result_reg, isLastTile);
                shuffle_penalty_I_row_behind(tileNr);
                r++;

                for(; r < (group_size) - 3; r += 4){
                    hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
                    if (!threadIdInGroup) hap_letter = new_hap_letter4.w;
                    state.relax_I_row_behind(r, tileNr, hap_letter, threadIdInWarp, threadIdInGroup, result_reg, isLastTile);
                    shuffle_penalty_I_row_behind(tileNr);

                    if(r/4 < numHaploChar4s){
                        new_hap_letter4 = HapsAsChar4[(r)/4];
                    }
                    hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
                    if (!threadIdInGroup) hap_letter = new_hap_letter4.x;
                    state.relax_I_row_behind(r+1, tileNr, hap_letter, threadIdInWarp, threadIdInGroup, result_reg, isLastTile);
                    shuffle_penalty_I_row_behind(tileNr);

                    hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
                    if (!threadIdInGroup) hap_letter = new_hap_letter4.y;
                    state.relax_I_row_behind(r+2, tileNr, hap_letter, threadIdInWarp, threadIdInGroup, result_reg, isLastTile);
                    shuffle_penalty_I_row_behind(tileNr);

                    hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
                    if (!threadIdInGroup) hap_letter = new_hap_letter4.z;
                    state.relax_I_row_behind(r+3, tileNr, hap_letter, threadIdInWarp, threadIdInGroup, result_reg, isLastTile);
                    shuffle_penalty_I_row_behind(tileNr);
                }

                //process rows which need to write right column
                for(; r < numRows - 3; r += 4){ 
                    hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
                    if (!threadIdInGroup) hap_letter = new_hap_letter4.w;
                    state.relax_I_row_behind(r, tileNr, hap_letter, threadIdInWarp, threadIdInGroup, result_reg, isLastTile);
                    shuffle_penalty_I_row_behind(tileNr);
                    cacheTileLastColumn();      

                    if(r/4 < numHaploChar4s){
                        new_hap_letter4 = HapsAsChar4[(r)/4];
                    }
                    hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
                    if (!threadIdInGroup) hap_letter = new_hap_letter4.x;
                    state.relax_I_row_behind(r+1, tileNr, hap_letter, threadIdInWarp, threadIdInGroup, result_reg, isLastTile);
                    shuffle_penalty_I_row_behind(tileNr);
                    cacheTileLastColumn();

                    hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
                    if (!threadIdInGroup) hap_letter = new_hap_letter4.y;
                    state.relax_I_row_behind(r+2, tileNr, hap_letter, threadIdInWarp, threadIdInGroup, result_reg, isLastTile);
                    shuffle_penalty_I_row_behind(tileNr);
                    cacheTileLastColumn();

                    hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
                    if (!threadIdInGroup) hap_letter = new_hap_letter4.z;
                    state.relax_I_row_behind(r+3, tileNr, hap_letter, threadIdInWarp, threadIdInGroup, result_reg, isLastTile);
                    shuffle_penalty_I_row_behind(tileNr);
                    cacheTileLastColumn(); 

                    if((r+4) % (group_size) == 0){
                        writeCachedTileLastColumnToMemory();
                    }
        
                }

                if(r < numRows){
                    hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
                    if (!threadIdInGroup) hap_letter = new_hap_letter4.w;
                    state.relax_I_row_behind(r, tileNr, hap_letter, threadIdInWarp, threadIdInGroup, result_reg, isLastTile);
                    shuffle_penalty_I_row_behind(tileNr);
                    cacheTileLastColumn();
                }
                if(r+1 < numRows){
                    if(r/4 < numHaploChar4s){
                        new_hap_letter4 = HapsAsChar4[(r)/4];
                    }
                    hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
                    if (!threadIdInGroup) hap_letter = new_hap_letter4.x;
                    state.relax_I_row_behind(r+1, tileNr, hap_letter, threadIdInWarp, threadIdInGroup, result_reg, isLastTile);
                    shuffle_penalty_I_row_behind(tileNr);
                    cacheTileLastColumn();
                }
                if(r+2 < numRows){
                    hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
                    if (!threadIdInGroup) hap_letter = new_hap_letter4.y;
                    state.relax_I_row_behind(r+2, tileNr, hap_letter, threadIdInWarp, threadIdInGroup, result_reg, isLastTile);
                    shuffle_penalty_I_row_behind(tileNr);
                    cacheTileLastColumn();
                }

                // if(r < numRows){
                //     hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
                //     if (!threadIdInGroup) hap_letter = new_hap_letter4.w;
                //     state.relax_I_row_behind(r, tileNr, hap_letter, threadIdInWarp, threadIdInGroup, result_reg, isLastTile);
                //     shuffle_penalty_I_row_behind(tileNr);
                //     cacheTileLastColumn();
                //     r++;
                // }
                // if(r < numRows){
                //     if((r-1)/4 < numHaploChar4s){
                //         new_hap_letter4 = HapsAsChar4[(r-1)/4];
                //     }
                //     hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
                //     if (!threadIdInGroup) hap_letter = new_hap_letter4.x;
                //     state.relax_I_row_behind(r, tileNr, hap_letter, threadIdInWarp, threadIdInGroup, result_reg, isLastTile);
                //     shuffle_penalty_I_row_behind(tileNr);
                //     cacheTileLastColumn();
                //     r++;
                // }
                // if(r < numRows){
                //     hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
                //     if (!threadIdInGroup) hap_letter = new_hap_letter4.y;
                //     state.relax_I_row_behind(r, tileNr, hap_letter, threadIdInWarp, threadIdInGroup, result_reg, isLastTile);
                //     shuffle_penalty_I_row_behind(tileNr);
                //     cacheTileLastColumn();
                //     r++;
                // }

                // Relaxation computes I for the previous diagonal. Thus, we are missing I for the last diagonal.
                // Compute it now!
                state.relax_final_I_row(numRows-1, tileNr, hap_letter, threadIdInWarp, threadIdInGroup, result_reg, isLastTile);
                cacheTileLastColumn();

                //write the remaining cached data to memory
                //since we do an extra step at the end to compute the final I, last column of haploLength+1 many rows were cached in total
                const int totalCachedRows = haploLength+1;
                const int cachedRowsNotWrittenToMemory = (totalCachedRows % group_size == 0) ? group_size : totalCachedRows % group_size;
                if(cachedRowsNotWrittenToMemory > 0){
                    const int firstValidThread = group_size - cachedRowsNotWrittenToMemory;
                    if(threadIdInGroup >= firstValidThread){
                        writeCachedTileLastColumnToMemoryFinal(firstValidThread);
                    }
                }
            }
        
            //process intermediate tiles
            for(int tileNr = 1; tileNr < maxNumTilesInWarp-1; tileNr++){
                {
                
                    constexpr bool isLastTile = false;
                    // constexpr bool coversRow0Yes = true;
                    // constexpr bool coversRow0No = false;
                    tempWriteOffset = threadIdInGroup;
                    tempLoadOffset = threadIdInGroup;
    
                    load_PSSM(tileNr);
                    load_probabilities(tileNr);
    
                    loadTileLeftBorderFromMemory();
                    init_penalties_notFirstTile(tileNr, tileLeftBorder_M, tileLeftBorder_I, tileLeftBorder_D);
    
                    const int numRows = haploLength + (group_size-1) + 1;
                    const int numHaploChar4s = SDIV(haploLength, 4);

                    char4 new_hap_letter4;
                    hap_letter = 4;
                    int r = 1;
    
                    // process first (group_size-1) diagonals for which some threads are out-of-bounds or in the pre-initialized row
                    new_hap_letter4 = HapsAsChar4[(r-1)/4];
                    hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
                    if (!threadIdInGroup) hap_letter = new_hap_letter4.x;
                    state.relax_I_row_behind(r, tileNr, hap_letter, threadIdInWarp, threadIdInGroup, result_reg, isLastTile);
                    leftBorderShuffleDown();
                    shuffle_penalty_I_row_behind_notFirstTile(tileNr, tileLeftBorder_M, tileLeftBorder_I, tileLeftBorder_D);
                    r++;
    
                    hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
                    if (!threadIdInGroup) hap_letter = new_hap_letter4.y;
                    state.relax_I_row_behind(r, tileNr, hap_letter, threadIdInWarp, threadIdInGroup, result_reg, isLastTile);
                    leftBorderShuffleDown();
                    shuffle_penalty_I_row_behind_notFirstTile(tileNr, tileLeftBorder_M, tileLeftBorder_I, tileLeftBorder_D);
                    r++;
    
                    hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
                    if (!threadIdInGroup) hap_letter = new_hap_letter4.z;
                    state.relax_I_row_behind(r, tileNr, hap_letter, threadIdInWarp, threadIdInGroup, result_reg, isLastTile);
                    leftBorderShuffleDown();
                    shuffle_penalty_I_row_behind_notFirstTile(tileNr, tileLeftBorder_M, tileLeftBorder_I, tileLeftBorder_D);
                    r++;
    
                    for(; r < (group_size) - 3; r += 4){
                        hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
                        if (!threadIdInGroup) hap_letter = new_hap_letter4.w;
                        state.relax_I_row_behind(r, tileNr, hap_letter, threadIdInWarp, threadIdInGroup, result_reg, isLastTile);
                        leftBorderShuffleDown();
                        shuffle_penalty_I_row_behind_notFirstTile(tileNr, tileLeftBorder_M, tileLeftBorder_I, tileLeftBorder_D);
    
                        if(r/4 < numHaploChar4s){
                            new_hap_letter4 = HapsAsChar4[(r)/4];
                        }
                        hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
                        if (!threadIdInGroup) hap_letter = new_hap_letter4.x;
                        state.relax_I_row_behind(r+1, tileNr, hap_letter, threadIdInWarp, threadIdInGroup, result_reg, isLastTile);
                        leftBorderShuffleDown();
                        shuffle_penalty_I_row_behind_notFirstTile(tileNr, tileLeftBorder_M, tileLeftBorder_I, tileLeftBorder_D);
    
                        hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
                        if (!threadIdInGroup) hap_letter = new_hap_letter4.y;
                        state.relax_I_row_behind(r+2, tileNr, hap_letter, threadIdInWarp, threadIdInGroup, result_reg, isLastTile);
                        leftBorderShuffleDown();
                        shuffle_penalty_I_row_behind_notFirstTile(tileNr, tileLeftBorder_M, tileLeftBorder_I, tileLeftBorder_D);
    
                        hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
                        if (!threadIdInGroup) hap_letter = new_hap_letter4.z;
                        state.relax_I_row_behind(r+3, tileNr, hap_letter, threadIdInWarp, threadIdInGroup, result_reg, isLastTile);
                        leftBorderShuffleDown();
                        shuffle_penalty_I_row_behind_notFirstTile(tileNr, tileLeftBorder_M, tileLeftBorder_I, tileLeftBorder_D);
                    }
    
                    //process rows which need to write right column
                    for(; r < numRows - 3; r += 4){ 
                        hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
                        if (!threadIdInGroup) hap_letter = new_hap_letter4.w;
                        state.relax_I_row_behind(r, tileNr, hap_letter, threadIdInWarp, threadIdInGroup, result_reg, isLastTile);
                        leftBorderShuffleDown();
                        shuffle_penalty_I_row_behind_notFirstTile(tileNr, tileLeftBorder_M, tileLeftBorder_I, tileLeftBorder_D);
                        cacheTileLastColumn();      
    
                        if(r/4 < numHaploChar4s){
                            new_hap_letter4 = HapsAsChar4[(r)/4];
                        }
                        hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
                        if (!threadIdInGroup) hap_letter = new_hap_letter4.x;

                        if(r % group_size == 0 && r <= haploLength+1){
                            loadTileLeftBorderFromMemory();
                            if(!threadIdInGroup){
                                updateStateFromLeftBorder(tileLeftBorder_M, tileLeftBorder_I, tileLeftBorder_D);
                            }
                        }

                        state.relax_I_row_behind(r+1, tileNr, hap_letter, threadIdInWarp, threadIdInGroup, result_reg, isLastTile);
                        leftBorderShuffleDown();
                        shuffle_penalty_I_row_behind_notFirstTile(tileNr, tileLeftBorder_M, tileLeftBorder_I, tileLeftBorder_D);
                        cacheTileLastColumn();
    
                        hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
                        if (!threadIdInGroup) hap_letter = new_hap_letter4.y;
                        state.relax_I_row_behind(r+2, tileNr, hap_letter, threadIdInWarp, threadIdInGroup, result_reg, isLastTile);
                        leftBorderShuffleDown();
                        shuffle_penalty_I_row_behind_notFirstTile(tileNr, tileLeftBorder_M, tileLeftBorder_I, tileLeftBorder_D);
                        cacheTileLastColumn();
    
                        hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
                        if (!threadIdInGroup) hap_letter = new_hap_letter4.z;
                        state.relax_I_row_behind(r+3, tileNr, hap_letter, threadIdInWarp, threadIdInGroup, result_reg, isLastTile);
                        leftBorderShuffleDown();
                        shuffle_penalty_I_row_behind_notFirstTile(tileNr, tileLeftBorder_M, tileLeftBorder_I, tileLeftBorder_D);
                        cacheTileLastColumn(); 
    
                        if((r+4) % (group_size) == 0){
                            writeCachedTileLastColumnToMemory();
                        }
            
                    }
    
                    if(r < numRows){
                        hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
                        if (!threadIdInGroup) hap_letter = new_hap_letter4.w;
                        state.relax_I_row_behind(r, tileNr, hap_letter, threadIdInWarp, threadIdInGroup, result_reg, isLastTile);
                        leftBorderShuffleDown();
                        shuffle_penalty_I_row_behind_notFirstTile(tileNr, tileLeftBorder_M, tileLeftBorder_I, tileLeftBorder_D);
                        cacheTileLastColumn();
                    }
                    if(r+1 < numRows){
                        if(r/4 < numHaploChar4s){
                            new_hap_letter4 = HapsAsChar4[(r)/4];
                        }
                        hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
                        if (!threadIdInGroup) hap_letter = new_hap_letter4.x;

                        if(r % group_size == 0 && r <= haploLength+1){
                            loadTileLeftBorderFromMemory();
                            if(!threadIdInGroup){
                                updateStateFromLeftBorder(tileLeftBorder_M, tileLeftBorder_I, tileLeftBorder_D);
                            }
                        }

                        state.relax_I_row_behind(r+1, tileNr, hap_letter, threadIdInWarp, threadIdInGroup, result_reg, isLastTile);
                        leftBorderShuffleDown();
                        shuffle_penalty_I_row_behind_notFirstTile(tileNr, tileLeftBorder_M, tileLeftBorder_I, tileLeftBorder_D);
                        cacheTileLastColumn();
                    }
                    if(r+2 < numRows){
                        hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
                        if (!threadIdInGroup) hap_letter = new_hap_letter4.y;
                        state.relax_I_row_behind(r+2, tileNr, hap_letter, threadIdInWarp, threadIdInGroup, result_reg, isLastTile);
                        leftBorderShuffleDown();
                        shuffle_penalty_I_row_behind_notFirstTile(tileNr, tileLeftBorder_M, tileLeftBorder_I, tileLeftBorder_D);
                        cacheTileLastColumn();
                    }
    
                    // Relaxation computes I for the previous diagonal. Thus, we are missing I for the last diagonal.
                    // Compute it now!

                    const int lastIRow = numRows-1;
                    //for the computation of row r (from view of thread 0), we would typically use (r-1) in the condition for border loading
                    //however, the value we need to load for final I calculation is stored one row ahead in memory.
                    //so check for lastIRow instead of lastIRow-1
                    if(lastIRow % group_size == 0 && lastIRow <= haploLength+1){
                        loadTileLeftBorderFromMemory();
                        if(!threadIdInGroup){
                            updateStateFromLeftBorder(tileLeftBorder_M, tileLeftBorder_I, tileLeftBorder_D);
                        }
                    }
                    state.relax_final_I_row(lastIRow, tileNr, hap_letter, threadIdInWarp, threadIdInGroup, result_reg, isLastTile);
                    cacheTileLastColumn();

                    //write the remaining cached data to memory
                    //since we do an extra step at the end to compute the final I, last column of haploLength+1 many rows were cached in total
                    const int totalCachedRows = haploLength+1;
                    const int cachedRowsNotWrittenToMemory = (totalCachedRows % group_size == 0) ? group_size : totalCachedRows % group_size;
                    if(cachedRowsNotWrittenToMemory > 0){
                        const int firstValidThread = group_size - cachedRowsNotWrittenToMemory;
                        if(threadIdInGroup >= firstValidThread){
                            writeCachedTileLastColumnToMemoryFinal(firstValidThread);
                        }
                    }
    
                }
            }

            //process last tile
            {
                const int tileNr = maxNumTilesInWarp-1;
                constexpr bool isLastTile = true;
                // constexpr bool coversRow0Yes = true;
                // constexpr bool coversRow0No = false;
                tempWriteOffset = threadIdInGroup;
                tempLoadOffset = threadIdInGroup;

                load_PSSM(tileNr);
                load_probabilities(tileNr);

                loadTileLeftBorderFromMemory();
                init_penalties_notFirstTile(tileNr, tileLeftBorder_M, tileLeftBorder_I, tileLeftBorder_D);

                //const int numRows = haploLength + group_size - 1;
                const int numRows = haploLength + result_thread + 1;
                const int numHaploChar4s = SDIV(haploLength, 4);
                char4 new_hap_letter4;
                hap_letter = 4;
                int r = 1;

                // process first (group_size-1) diagonals for which some threads are out-of-bounds or in the pre-initialized row
                if(r < numRows){
                    new_hap_letter4 = HapsAsChar4[(r-1)/4];
                    hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
                    if (!threadIdInGroup) hap_letter = new_hap_letter4.x;
                    state.relax_I_row_behind(r, tileNr, hap_letter, threadIdInWarp, threadIdInGroup, result_reg, isLastTile);
                    leftBorderShuffleDown();
                    shuffle_penalty_I_row_behind_notFirstTile(tileNr, tileLeftBorder_M, tileLeftBorder_I, tileLeftBorder_D);
                    r++;
                }

                if(r < numRows){
                    hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
                    if (!threadIdInGroup) hap_letter = new_hap_letter4.y;
                    state.relax_I_row_behind(r, tileNr, hap_letter, threadIdInWarp, threadIdInGroup, result_reg, isLastTile);
                    leftBorderShuffleDown();
                    shuffle_penalty_I_row_behind_notFirstTile(tileNr, tileLeftBorder_M, tileLeftBorder_I, tileLeftBorder_D);
                    r++;
                }

                if(r < numRows){
                    hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
                    if (!threadIdInGroup) hap_letter = new_hap_letter4.z;
                    state.relax_I_row_behind(r, tileNr, hap_letter, threadIdInWarp, threadIdInGroup, result_reg, isLastTile);
                    leftBorderShuffleDown();
                    shuffle_penalty_I_row_behind_notFirstTile(tileNr, tileLeftBorder_M, tileLeftBorder_I, tileLeftBorder_D);
                    r++;
                }

                for(; r < min(group_size, numRows);){
                    if(r < numRows){
                        hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
                        if (!threadIdInGroup) hap_letter = new_hap_letter4.w;
                        state.relax_I_row_behind(r, tileNr, hap_letter, threadIdInWarp, threadIdInGroup, result_reg, isLastTile);
                        leftBorderShuffleDown();
                        shuffle_penalty_I_row_behind_notFirstTile(tileNr, tileLeftBorder_M, tileLeftBorder_I, tileLeftBorder_D);
                        r++;
                    }

                    if(r < numRows){
                        if((r-1)/4 < numHaploChar4s){
                            new_hap_letter4 = HapsAsChar4[(r-1)/4];
                        }
                        hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
                        if (!threadIdInGroup) hap_letter = new_hap_letter4.x;
                        state.relax_I_row_behind(r, tileNr, hap_letter, threadIdInWarp, threadIdInGroup, result_reg, isLastTile);
                        leftBorderShuffleDown();
                        shuffle_penalty_I_row_behind_notFirstTile(tileNr, tileLeftBorder_M, tileLeftBorder_I, tileLeftBorder_D);
                        r++;
                    }

                    if(r < numRows){
                        hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
                        if (!threadIdInGroup) hap_letter = new_hap_letter4.y;
                        state.relax_I_row_behind(r, tileNr, hap_letter, threadIdInWarp, threadIdInGroup, result_reg, isLastTile);
                        leftBorderShuffleDown();
                        shuffle_penalty_I_row_behind_notFirstTile(tileNr, tileLeftBorder_M, tileLeftBorder_I, tileLeftBorder_D);
                        r++;
                    }

                    if(r < numRows){
                        hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
                        if (!threadIdInGroup) hap_letter = new_hap_letter4.z;
                        state.relax_I_row_behind(r, tileNr, hap_letter, threadIdInWarp, threadIdInGroup, result_reg, isLastTile);
                        leftBorderShuffleDown();
                        shuffle_penalty_I_row_behind_notFirstTile(tileNr, tileLeftBorder_M, tileLeftBorder_I, tileLeftBorder_D);
                        r++;
                    }
                }

                //process rows which need to write right column
                for(; r < numRows - 3; r += 4){ 
                    hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
                    if (!threadIdInGroup) hap_letter = new_hap_letter4.w;
                    state.relax_I_row_behind(r, tileNr, hap_letter, threadIdInWarp, threadIdInGroup, result_reg, isLastTile);
                    leftBorderShuffleDown();
                    shuffle_penalty_I_row_behind_notFirstTile(tileNr, tileLeftBorder_M, tileLeftBorder_I, tileLeftBorder_D);     

                    if(r/4 < numHaploChar4s){
                        new_hap_letter4 = HapsAsChar4[(r)/4];
                    }
                    hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
                    if (!threadIdInGroup) hap_letter = new_hap_letter4.x;

                    if(r % group_size == 0 && r <= haploLength+1){
                        loadTileLeftBorderFromMemory();
                        if(!threadIdInGroup){
                            updateStateFromLeftBorder(tileLeftBorder_M, tileLeftBorder_I, tileLeftBorder_D);
                        }
                    }

                    state.relax_I_row_behind(r+1, tileNr, hap_letter, threadIdInWarp, threadIdInGroup, result_reg, isLastTile);
                    leftBorderShuffleDown();
                    shuffle_penalty_I_row_behind_notFirstTile(tileNr, tileLeftBorder_M, tileLeftBorder_I, tileLeftBorder_D);

                    hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
                    if (!threadIdInGroup) hap_letter = new_hap_letter4.y;
                    state.relax_I_row_behind(r+2, tileNr, hap_letter, threadIdInWarp, threadIdInGroup, result_reg, isLastTile);
                    leftBorderShuffleDown();
                    shuffle_penalty_I_row_behind_notFirstTile(tileNr, tileLeftBorder_M, tileLeftBorder_I, tileLeftBorder_D);

                    hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
                    if (!threadIdInGroup) hap_letter = new_hap_letter4.z;
                    state.relax_I_row_behind(r+3, tileNr, hap_letter, threadIdInWarp, threadIdInGroup, result_reg, isLastTile);
                    leftBorderShuffleDown();
                    shuffle_penalty_I_row_behind_notFirstTile(tileNr, tileLeftBorder_M, tileLeftBorder_I, tileLeftBorder_D);        
                }

                if(r < numRows){
                    hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
                    if (!threadIdInGroup) hap_letter = new_hap_letter4.w;
                    state.relax_I_row_behind(r, tileNr, hap_letter, threadIdInWarp, threadIdInGroup, result_reg, isLastTile);
                    leftBorderShuffleDown();
                    shuffle_penalty_I_row_behind_notFirstTile(tileNr, tileLeftBorder_M, tileLeftBorder_I, tileLeftBorder_D);
                }
                if(r+1 < numRows){
                    if(r/4 < numHaploChar4s){
                        new_hap_letter4 = HapsAsChar4[(r)/4];
                    }
                    hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
                    if (!threadIdInGroup) hap_letter = new_hap_letter4.x;

                    if(r % group_size == 0 && r <= haploLength+1){
                        loadTileLeftBorderFromMemory();
                        if(!threadIdInGroup){
                            updateStateFromLeftBorder(tileLeftBorder_M, tileLeftBorder_I, tileLeftBorder_D);
                        }
                    }

                    state.relax_I_row_behind(r+1, tileNr, hap_letter, threadIdInWarp, threadIdInGroup, result_reg, isLastTile);
                    leftBorderShuffleDown();
                    shuffle_penalty_I_row_behind_notFirstTile(tileNr, tileLeftBorder_M, tileLeftBorder_I, tileLeftBorder_D);
                }
                if(r+2 < numRows){
                    hap_letter = __shfl_up_sync(myGroupMask, hap_letter, 1, group_size);
                    if (!threadIdInGroup) hap_letter = new_hap_letter4.y;
                    state.relax_I_row_behind(r+2, tileNr, hap_letter, threadIdInWarp, threadIdInGroup, result_reg, isLastTile);
                    leftBorderShuffleDown();
                    shuffle_penalty_I_row_behind_notFirstTile(tileNr, tileLeftBorder_M, tileLeftBorder_I, tileLeftBorder_D);
                }

                // Relaxation computes I for the previous diagonal. Thus, we are missing I for the last diagonal.
                // Compute it now!

                const int lastIRow = numRows-1;
                //for the computation of row r (from view of thread 0), we would typically use (r-1) in the condition for border loading
                //however, the value we need to load for final I calculation is stored one row ahead in memory.
                //so check for lastIRow instead of lastIRow-1
                if(lastIRow % group_size == 0 && lastIRow <= haploLength+1){
                    loadTileLeftBorderFromMemory();
                    if(!threadIdInGroup){
                        updateStateFromLeftBorder(tileLeftBorder_M, tileLeftBorder_I, tileLeftBorder_D);
                    }
                }
                state.relax_final_I_row(lastIRow, tileNr, hap_letter, threadIdInWarp, threadIdInGroup, result_reg, isLastTile);
            }

            if (threadIdInGroup == result_thread) {
                float temp_res = state.myResult;
                temp_res =  log10f(temp_res) - log10f(constant);
                resultoutput[resultOutputIndex] = temp_res;
            }

        }

        

        
    }


}



template <int group_size, int numRegs> 
void call_PairHMM_float_multitile_kernel(
    char* d_temp,
    size_t tempBytes,
    int maximumHaplotypeLength,
    float* d_resultoutput,
    const uint8_t* d_readData,
    const uint8_t* d_hapData,
    const uint8_t* d_base_quals,
    const uint8_t* d_ins_quals,
    const uint8_t* d_del_quals,
    const int* readBeginOffsets,
    const int* hapBeginOffsets,
    const int* d_readLengths,
    const int* d_hapLengths,
    const int* d_numHapsPerSequenceGroup,
    const int* d_numHapsPerSequenceGroupPrefixSum,
    const int* d_indicesPerSequenceGroup,
    const int* d_numReadsPerSequenceGroupPrefixSum,
    const int numSequenceGroups,
    const int* d_resultOffsetsPerSequenceGroup,
    const int* d_numAlignmentsPerSequenceGroup,
    const int* d_numAlignmentsPerSequenceGroupInclusivePrefixSum,
    const int numAlignments,
    cudaStream_t stream
){
    if(d_temp == nullptr || tempBytes == 0) throw std::runtime_error("tempstorage is 0");

    auto kernel = PairHMM_float_multitile_kernel<group_size, numRegs>;
    constexpr int blocksize = 32;
    int deviceId = 0;
    int numSMs = 0;
    cudaGetDevice(&deviceId);
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, deviceId);
    int smem = 0;
    int maxBlocksPerSM = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxBlocksPerSM,
        kernel,
        blocksize, 
        smem
    );
    const size_t tileTempBytesPerGroup = sizeof(float) * 3 * (maximumHaplotypeLength + 1 + group_size);

    constexpr int groupsPerBlock = (blocksize / group_size);
    constexpr int alignmentsPerBlock = groupsPerBlock;
    const int maxNumBlocksByInputSize = (numAlignments + alignmentsPerBlock - 1) / alignmentsPerBlock;
    const int maxNumBlocksByOccupancy = maxBlocksPerSM * numSMs;
    const int maxNumBlocksByTempBytes = tempBytes / (tileTempBytesPerGroup * groupsPerBlock);

    const int numBlocks = std::min(maxNumBlocksByTempBytes, std::min(maxNumBlocksByInputSize, maxNumBlocksByOccupancy));
    if(numBlocks <= 0){
        throw std::runtime_error("could not launch kernel. numBlocks <= 0");
    }

    // std::cout << "numSMs: " << numSMs << ", numBlocks " << numBlocks << "\n";

    kernel<<<numBlocks, blocksize, 0, stream>>>(
        d_resultoutput,
        d_readData,
        d_hapData,
        d_base_quals,
        d_ins_quals,
        d_del_quals,
        readBeginOffsets,
        hapBeginOffsets,
        d_readLengths,
        d_hapLengths,
        d_numHapsPerSequenceGroup,
        d_numHapsPerSequenceGroupPrefixSum,
        d_indicesPerSequenceGroup,
        d_numReadsPerSequenceGroupPrefixSum,
        numSequenceGroups,
        d_resultOffsetsPerSequenceGroup,
        d_numAlignmentsPerSequenceGroup,
        d_numAlignmentsPerSequenceGroupInclusivePrefixSum,
        numAlignments,
        d_temp,
        tileTempBytesPerGroup,
        maximumHaplotypeLength
    );
}


#endif