const sampler_t sampler =  CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
/************************************ get mean ************************************/
__kernel void get_mean(
    read_only image2d_t tgt,
    read_only image2d_t src,
    write_only image2d_t tgt_mean,
    write_only image2d_t src_mean,
    const int kernel_size)
{
    const int col = get_global_id(0);
    const int row = get_global_id(1);
    int2 pos = (int2)(col, row);

    int mSize = kernel_size / 2;

    float4 tmp_tgt = (float4)(0.f);
    float4 tmp_src = (float4)(0.f);
    for(int i=-mSize; i<=mSize; i++)
        for(int j=-mSize; j<=mSize; j++)
        {
            int2 tmp_pos = (int2)(col + j, row + i);
            tmp_tgt += read_imagef(tgt, sampler, tmp_pos);
            tmp_src += read_imagef(src, sampler, tmp_pos);
        }
    tmp_tgt.w = 0.f;
    tmp_src.w = 0.f;
    write_imagef(tgt_mean, pos, tmp_tgt / (kernel_size * kernel_size)); // automatically converted to float4 data type
    write_imagef(src_mean, pos, tmp_src / (kernel_size * kernel_size));
}

/************************************ get cost ************************************/
__kernel void get_cost(
    read_only image2d_t tgt,
    read_only image2d_t src,
    read_only image2d_t tgt_mean,
    read_only image2d_t src_mean,
    __global float* cost,
    const int width,
    const int kernel_size,
    const int dispRange)
{
    const int col = get_global_id(0);
    const int row = get_global_id(1);
    const int depth = get_global_id(2);

    if(col - depth < 0) return;

    const int idx = dispRange * (width * row + col) + depth;

    int2 pos = (int2)(col, row);

    int mSize = kernel_size / 2;

    float4 tmpA = (float4)(0.f);
    float4 tmpB = (float4)(0.f);
    float4 tmpC = (float4)(0.f);
    for(int i=-mSize; i<=mSize; i++)
        for(int j=-mSize; j<=mSize; j++)
        {
            int2 tgt_pos = (int2)(col + j, row + i);
            int2 src_pos = (int2)(col + j - depth, row + i);
            float4 A_ = read_imagef(tgt, sampler, tgt_pos) - read_imagef(tgt_mean, sampler, tgt_pos);
            float4 B_ = read_imagef(src, sampler, src_pos) - read_imagef(src_mean, sampler, src_pos);
            tmpA += (A_ * B_);
            tmpB += (A_ * A_);
            tmpC += (B_ * B_);
        }
    float4 tmp = tmpA / sqrt(tmpB * tmpC);
    cost[idx] = (tmp.x + tmp.y + tmp.z) / 3;
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
	
	float maxCost = -100000.f;
	int maxIndex = 0;

	for (int d = 0; d < dispRange; d++) {
		float tmp = datacost[dispRange * (width * row + col) + d];
		if (maxCost < tmp)
		{
			maxCost = tmp;
			maxIndex = d;
		}
	}

	result[width * row + col] = (uchar)(maxIndex);
}

__kernel void wta2(
    __global float4* cost,
    __global uchar* res,
    const int width,
    const int dispRange)
{
    const int col = get_global_id(0);
    const int row = get_global_id(1);

    float maxCost = -100000.f;
	int maxIndex = 0;

    for (int d = 0; d < dispRange; d++) {
		int lock_d = d / 4;

		float tmp = cost[dispRange * (width * row + col) / 4 + lock_d].x;
		if (maxCost < tmp)
		{
			maxCost = tmp;
			maxIndex = d;
		}
		d+=1;

		tmp = cost[dispRange * (width * row + col) / 4 + lock_d].y;
		if (maxCost < tmp)
		{
			maxCost = tmp;
			maxIndex = d;
		}
		d+=1;

		tmp = cost[dispRange * (width * row + col) / 4 + lock_d].z;
		if (maxCost < tmp)
		{
			maxCost = tmp;
			maxIndex = d;
		}
		d+=1;

		tmp = cost[dispRange * (width * row + col) / 4 + lock_d].w;
		if (maxCost < tmp)
		{
			maxCost = tmp;
			maxIndex = d;
		}
	}

	res[width * row + col] = (uchar)(maxIndex);
}