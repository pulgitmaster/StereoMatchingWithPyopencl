const sampler_t sampler =  CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

/************************************ get cost ************************************/
__kernel void get_cost(
    read_only image2d_t tgt,
    read_only image2d_t src,
    __global float* cost,
    const int width,
    const int kernel_size,
    const int dispRange)
{
    const int col = get_global_id(0);
    const int row = get_global_id(1);
    const int depth = get_global_id(2);

    const int idx = dispRange * (width * row + col) + depth;

    if(col - depth < 0)
	{
		cost[idx] = 1000000.f;
		return;
	}

    int2 pos = (int2)(col, row);

    int mSize = kernel_size / 2;

    float4 diff = (float4)(0);
    for(int i=-mSize; i<=mSize; i++)
        for(int j=-mSize; j<=mSize; j++)
        {
            int2 tgt_pos = (int2)(col + j, row + i);
            int2 src_pos = (int2)(col + j - depth, row + i);
			float4 tmpA = read_imagef(tgt, sampler, tgt_pos);
			float4 tmpB = read_imagef(src, sampler, src_pos);
            diff += fabs(tmpA - tmpB);
        }
    cost[idx] = (diff.x + diff.y + diff.z) / (float)(kernel_size * kernel_size);
}

/************************************ winner takes all ************************************/
__kernel void wta(
    __global float* datacost,
    __global uchar* result,
	const int width, 
    const int dispRange)
{
	int col = get_global_id(0);
	int row = get_global_id(1);
	
	float minCost = 100000.f;
	int minIndex = 0;

	for (int d = 0; d < dispRange; d++) {
		float tmp = datacost[dispRange * (width * row + col) + d];
		if (minCost > tmp)
		{
			minCost = tmp;
			minIndex = d;
		}
	}

	result[width * row + col] = (uchar)(minIndex);
}

__kernel void wta2(
    __global float4* cost,
    __global uchar* res,
    const int width,
    const int dispRange)
{
    const int col = get_global_id(0);
    const int row = get_global_id(1);

    float minCost = 100000.f;
	int minIndex = 0;

    for (int d = 0; d < dispRange; d++) {
		int lock_d = d / 4;

		float tmp = cost[dispRange * (width * row + col) / 4 + lock_d].x;
		if (minCost > tmp)
		{
			minCost = tmp;
			minIndex = d;
		}
		d+=1;

		tmp = cost[dispRange * (width * row + col) / 4 + lock_d].y;
		if (minCost > tmp)
		{
			minCost = tmp;
			minIndex = d;
		}
		d+=1;

		tmp = cost[dispRange * (width * row + col) / 4 + lock_d].z;
		if (minCost > tmp)
		{
			minCost = tmp;
			minIndex = d;
		}
		d+=1;

		tmp = cost[dispRange * (width * row + col) / 4 + lock_d].w;
		if (minCost > tmp)
		{
			minCost = tmp;
			minIndex = d;
		}
	}

	res[width * row + col] = (uchar)(minIndex);
}