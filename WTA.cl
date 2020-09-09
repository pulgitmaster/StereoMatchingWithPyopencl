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