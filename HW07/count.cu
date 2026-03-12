#include "count.cuh"
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/execution_policy.h>

void count(const thrust::device_vector<int>& d_in,
           thrust::device_vector<int>& values,
           thrust::device_vector<int>& counts) 
{
    // Make a copy since d_in is const and we need to sort it
    thrust::device_vector<int> d_copy = d_in;
    
    // Sort the copy to group identical elements
    thrust::sort(d_copy.begin(), d_copy.end());

    // Resize output vectors to the maximum possible size (all unique)
    values.resize(d_copy.size());
    counts.resize(d_copy.size());

    // Use reduce_by_key to count occurrences
    // The keys are the sorted numbers, the values to reduce are just 1s
    auto new_end = thrust::reduce_by_key(
        d_copy.begin(), d_copy.end(),
        thrust::constant_iterator<int>(1),
        values.begin(),
        counts.begin()
    );

    // Calculate actual number of unique elements and shrink output arrays
    int num_unique = new_end.first - values.begin();
    values.resize(num_unique);
    counts.resize(num_unique);
}
