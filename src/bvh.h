#include <float.h> // for max and min

//this is all heavily based on http://raytracey.blogspot.com/2016/01/gpu-path-tracing-tutorial-3-take-your.html

// bvh interface on host
struct BVHnode{
    float3 min;
    float3 max;
    virtual bool is_leaf() = 0;
};

// intermediate node. points to left and right
struct BVHinner: BVHnode {
    BVHnode* left;
    BVHnode* right;
    virtual bool is_leaf(){return false;}
};

// leaf node in tree. contains list of triangles
struct BVHleaf: BVHnode{
    std::vector<Triangle> triangles;
    virtual bool is_leaf(){return true;}
};

// triangles that haven't been added to bvh yet 
struct BBoxTemp{
    float3 min;
    float3 max;
    float3 center;
    Triangle triangle;
    BBoxTemp() :
	min({FLT_MAX, FLT_MAX, FLT_MAX}),
	max({-FLT_MAX, -FLT_MAX, -FLT_MAX})
    {}
};

// recursively build bvh split on planes parallel to axes
// tries to keep surface area the same for both halves
BVHnode* recurse(std::vector<BBoxTemp> working, int depth = 0){
    if (working.size() < 4){ // if only 4 triangles left
	BVHleaf* leaf = new BVHleaf;
	for(int i = 0; i< working.size(); ++i)
	    leaf->triangles.push_back(working[i].triangle);
	return leaf;	
    }
    float3 min = {FLT_MAX,FLT_MAX,FLT_MAX};
    float3 max = {-FLT_MAX,-FLT_MAX,-FLT_MAX};

    // calculate bounds for current working list
    for(uint i = 0; i<working.size(); ++i){
	BBoxTemp& v = working[i];
	min = minf3(min, v.min);
	max = maxf3(max, v.max);
    }

    //approxomate SA of triangle by size of bounding box
    float side1 = max.x - min.x;
    float side2 = max.y - min.y;
    float side3 = max.z - min.z;

    float min_cost = working.size() * (side1*side2 +
				      side2*side3 +
				      side3*side1);
    float best_split = FLT_MAX; // best value along axis

    int best_axis = -1; // best axis

    // 0 = X-axis, 1 = Y-axis, 2=Z-axis
    for(int i = 0; i< 3; ++i){ //check all three axes
	int axis = i;
	float start, stop, step;
	if(axis == 0){
	    start = min.x;
	    stop = max.x;
	}
	else if (axis == 1){
	    start = min.y;
	    stop = max.y;
	}
	else{
	    start = min.z;
	    stop = max.z;
	}

	// if box is too thin in this direction
	if (fabsf(stop - start) < 1e-4)
	    continue;

	// check discrete number of different splits on each axis
	// number gets smaller as we get farther into bvh and presumably smaller differences
	step = (stop - start) / (1024.0 / (depth + 1));

	// determine how good each plane is for splitting
	for(float test_split = start + step; test_split < stop-step; test_split += step){
	    float3 lmin = {FLT_MAX,FLT_MAX,FLT_MAX};
	    float3 lmax = {-FLT_MAX,-FLT_MAX,-FLT_MAX};

	    float3 rmin = {FLT_MAX,FLT_MAX,FLT_MAX};
	    float3 rmax = {-FLT_MAX,-FLT_MAX,-FLT_MAX};

	    int lcount = 0;
	    int rcount = 0;

	    for(uint j = 0; j<working.size(); ++j){
		BBoxTemp& v = working[j];
		float val;
		// use triangle center to determine which side to put it in
		if (axis == 0) val = v.center.x;
		else if (axis == 1) val = v.center.y;
		else val = v.center.z;

		if(val < test_split){
		    lmin = minf3(lmin, v.min);
		    lmax = maxf3(lmax, v.max);
		    lcount++;
		}
		else{
		    rmin = minf3(rmin, v.min);
		    rmax = maxf3(rmax, v.max);
		    rcount++;
		}
	    }

	    if (lcount <= 1 || rcount <=1) continue;

	    float lside1 = lmax.x - lmin.x;
	    float lside2 = lmax.y - lmin.y;
	    float lside3 = lmax.z - lmin.z;

	    float rside1 = rmax.x - rmin.x;
	    float rside2 = rmax.y - rmin.y;
	    float rside3 = rmax.z - rmin.z;

	    float lsurface = lside1*lside2 + lside2*lside3 + lside3*lside1;
	    float rsurface = rside1*rside2 + rside2*rside3 + rside3*rside1;

	    float total_cost =  lsurface*lcount + rsurface*rcount;
	    if (total_cost < min_cost){ // if this split is better, update stuff
		min_cost = total_cost;
		best_split = test_split;
		best_axis = axis;
	    }
	}
    }
    // if no split is better, just add a leaf node
    if (best_axis == -1){
	BVHleaf* leaf = new BVHleaf;
	for(int i = 0; i< working.size(); ++i)
	    leaf->triangles.push_back(working[i].triangle);
	return leaf;
    }

    // otherwise, create left and right working lists and call function recursively
    std::vector<BBoxTemp> left;
    std::vector<BBoxTemp> right;
    float3 lmin = {FLT_MAX,FLT_MAX,FLT_MAX};
    float3 lmax = {-FLT_MAX,-FLT_MAX,-FLT_MAX};
    float3 rmin = {FLT_MAX,FLT_MAX,FLT_MAX};
    float3 rmax = {-FLT_MAX,-FLT_MAX,-FLT_MAX};

    for(uint i = 0; i<working.size(); ++i){
	BBoxTemp& v = working[i];
	float val;
	if (best_axis == 0) val = v.center.x;
	else if (best_axis == 1) val = v.center.y;
	else val = v.center.z;
	if(val < best_split){
	    left.push_back(v);
	    lmin = minf3(lmin, v.min);
	    lmax = maxf3(lmax, v.max);
	}
	else{
	    right.push_back(v);
	    rmin = minf3(rmin, v.min);
	    rmax = maxf3(rmax, v.max);
	}
    }

    //create left and right child nodes
    BVHinner* inner = new BVHinner;
    inner->left = recurse(left, depth+1);
    inner->left->min = lmin;
    inner->left->max = lmax;

    inner->right = recurse(right, depth+1);
    inner->right->min = rmin;
    inner->right->max = rmax;

    return inner;
}

// create the host naive bvh
BVHnode* create_BVH(){
    std::vector<BBoxTemp> working;
    float3 min={FLT_MAX, FLT_MAX, FLT_MAX};
    float3 max={-FLT_MAX, -FLT_MAX, -FLT_MAX};

    std::cout << "Gathering box info..." << std::endl;
    
    for (uint i = 0; i<h_triangles.size(); ++i){
	const Triangle& triangle = h_triangles[i];

	BBoxTemp b;
	b.triangle = triangle;

	b.min = minf3(b.min, triangle.p);
	b.min = minf3(b.min, triangle.p + triangle.v0);
	b.min = minf3(b.min, triangle.p + triangle.v1);

	b.max = maxf3(b.max, triangle.p);
	b.max = maxf3(b.max, triangle.p + triangle.v0);
	b.max = maxf3(b.max, triangle.p + triangle.v1);

	min = minf3(min, b.min);
	max = maxf3(max, b.max);

	b.center = (b.max + b.min) * 0.5;

	working.push_back(b);
    }
    
    std::cout << "Creating BVH..." << std::endl;

    BVHnode* root = recurse(working);
    root->min = min;
    root->max = max;
    return root;
}

// each node takes up 32 bytes to allign nicely in memory
struct compact_BVHnode{
    float3 min;
    float3 max;
    union{
	struct{
	    uint left;
	    uint right;
	} inner;
	struct{
	    uint count;
	    uint offset;
	} leaf;
    }u;
};

std::vector<Triangle> triangles_ordered;
std::vector<compact_BVHnode> compact_BVH;

// convert naive bvh to memory friendly bvh
void populate_compact_BVHnode(const Triangle* first,
			      BVHnode* root,
			      uint& boxoffset,
			      uint& trioffset){
    int curr = compact_BVH.size();
    compact_BVHnode new_node;
    new_node.min = root->min;
    new_node.max = root->max;
    compact_BVH.push_back(new_node);
    if(!root->is_leaf()){
	BVHinner* p = dynamic_cast<BVHinner*>(root);
	int loffset = ++boxoffset;
	populate_compact_BVHnode(first, p->left, boxoffset, trioffset);
	int roffset = ++boxoffset;
	populate_compact_BVHnode(first, p->right, boxoffset, trioffset);
	compact_BVH[curr].u.inner.left = loffset;
	compact_BVH[curr].u.inner.right = roffset;
	//std::cout << loffset << " " << roffset << std::endl;
    }
    else{
	BVHleaf* p = dynamic_cast<BVHleaf*>(root);
	uint count = (uint)p->triangles.size();
	compact_BVH[curr].u.leaf.count = 0x80000000 | count;
	// use highest bit to indicate type of node because polymorphism is bad on gpu
	compact_BVH[curr].u.leaf.offset = trioffset;
	for(int i = 0; i<p->triangles.size(); ++i){
	    triangles_ordered.push_back(p->triangles[i]);
	    trioffset++;
	}
    }
}

void create_compact_BVH(){
    BVHnode* root = create_BVH();
    uint trioffset = 0;
    uint boxoffset = 0;
    populate_compact_BVHnode(&(h_triangles[0]), root, boxoffset, trioffset);
}
