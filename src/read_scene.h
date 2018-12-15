
// read file and add triangles to list
void create_mesh(std::string filename, float scale, float3 translate, scatter_type s,
		 texture_type t, int text_num, int meshes){
    const char* edot = strrchr(filename.c_str(), '.');
    std::vector<float3> verts;
    if (edot){
	edot++;
	//can read .ply files but assume in ascii not binary format
	if (!strcmp(edot, "PLY") || !strcmp(edot, "ply")){
	    std::ifstream file(filename, std::ios::in);
	    if (!file)
		print_error("Unable to find file: " + filename);
	    std::string line;
	    uint num_verts, num_tris;
	    bool in_header = true;
	    while (getline(file, line)){
		//first get all header info so we know how many verts and tris
		if (in_header){
		    if (line.substr(0,14) == "element vertex"){
			std::istringstream str(line);
			std::string w;
			str >> w;
			str >> w;
			str >> num_verts;
		    }
		    else if (line.substr(0, 12) == "element face"){
			std::istringstream str(line);
			std::string w;
			str >> w;
			str >> w;
			str >> num_tris;
		    }
		    else if (line.substr(0, 10) == "end_header")
			in_header = false;
		}
		else{ // read all verts
		    if (num_verts){
			num_verts--;
			float x, y, z;
			std::istringstream str_in(line);
			str_in >> x >> y >> z;
			verts.push_back({x,y,z});
		    }
		    else if (num_tris){ // once we have all verts, read triangles
			num_tris--;
			uint dummy, idx1, idx2, idx3;
			std::istringstream str(line);
			str >> dummy >> idx1 >> idx2 >> idx3;
			float3 v0 = scale*verts[idx1] + translate;
			float3 v1 = scale*verts[idx2] + translate;
			float3 v2 = scale*verts[idx3] + translate;
			h_triangles.push_back({v0, v1-v0, v2-v0,
				    normalize(cross(v1-v0, v2-v0)), s, t,
				    text_num, {0,0,0}, meshes});
		    }
		}
	    }
	}
	else if (!strcmp(edot, "obj")){ // can read .obj files but only ones with v/t/n format
	    std::ifstream file(filename, std::ios::in);
	    if (!file)
		print_error("Unable to find file: " + filename);
	    std::string line;
	    std::vector<float3> verts;
	    std::vector<float2> uv;
	    while(getline(file, line)){
		if (line.substr(0,2) == "v "){ // read vertex coordinate
		    float x, y, z;
		    sscanf(line.c_str(), "v %f %f %f", &x, &y, &z);
		    verts.push_back({x,y,z});
		}
		else if(line.substr(0,2) == "vt"){ //read uv for texture
		    float u, v;
		    sscanf(line.c_str(), "vt %f %f", &u, &v);
		    uv.push_back({u,v});
		}
		else if(line.substr(0,1) == "f"){ // read a face. assume triangle and v/t/n format
		    int v1, v2, v3, n1, n2, n3, t1, t2, t3;
		    sscanf(line.c_str(), "f %d/%d/%d %d/%d/%d %d/%d/%d",
			   &v1, &t1, &n1, &v2, &t2, &n2, &v3, &t3, &n3);
		    //std::cout << uv.size() <<  " " << t1 << std::endl;
		    float3 vert0 = scale*verts[v1-1] + translate;
		    float3 vert1 = scale*verts[v2-1] + translate;
		    float3 vert2 = scale*verts[v3-1] + translate;
		    float2 uv0 = uv[t1-1];
		    float2 uv1 = uv[t2-1];
		    float2 uv2 = uv[t3-1];
		    h_triangles.push_back({vert0, vert1-vert0, vert2-vert0,
				normalize(cross(vert1-vert0, vert2-vert0)), s, t,
				text_num, {uv0, uv1, uv2}, meshes});
		}
	    }
	}
	else{
	    print_error("Unknown file extension: " + std::string(edot)); 
	}
    }
    else{
	print_error("File " + filename + " has no extension");
    }
}

// take a file containing a texture and read it into memory
void create_texture(std::string filename){
    const char* edot = strrchr(filename.c_str(), '.');
    if(edot){
	edot++;
	if(!strcmp(edot, "png")){
	    uint w, h;
	    std::vector<unsigned char> image;
	    lodepng::decode(image, w, h, filename);
	    h_pix.push_back({(int)w,(int)h,(int)h_textures.size()});
	    for (int y = h-1; y >= 0; y--){
		for (int x = 0; x < w; x++){
		    int idx = 4*(y*w+x);
		    float r = float(image[idx])/255;
		    float g = float(image[idx + 1])/255;
		    float b = float(image[idx + 2])/255;
		    h_textures.push_back({r,g,b});
		}
	    }
	}
	else
	    print_error("Unknown file extension: " + std::string(edot));
    }
    else
	print_error("File " + filename + " has no extension");
}

// helper funtion to convert string from input into scatter and texture enums
void get_scat(char* scat, char* text, scatter_type& s, texture_type& t, int line_num){
    if (!strcmp(scat, "DIFF")) s = DIFF;
    else if (!strcmp(scat, "SPEC")) s = SPEC;
    else if (!strcmp(scat, "REFR")) s = REFR;
    else if (!strcmp(scat, "METAL")) s = METAL;
    else if (!strcmp(scat, "COAT")) s = COAT;
    else if (!strcmp(scat, "VOLUME")) s = VOLUME;
    else print_error("Invalid scatter type on line: " + std::to_string(line_num));
    
    if (!strcmp(text, "CONST")) t = CONST;
    else if (!strcmp(text, "CHECK")) t = CHECK;
    else if (!strcmp(text, "PIX")) t = PIX;
    else print_error("Invalid texture type on line: " + std::to_string(line_num));
}

// check to make sure enough things were read
// otherwise theres a problem in the config file
void check_read(int should_read, int actual_read, int line_num){
    if (should_read != actual_read)
	print_error("Couldn't read line " + std::to_string(line_num));
}

void read_scene(){
    std::ifstream file("scene", std::ios::in);
    std::string line;
    if (!file)
	print_error("Unable to find scene file");
    int line_num = 0;
    int read_num;
    int meshes = 0;
    while(getline(file, line)){
	line_num++;
	// dont hard code width, hight, samples per pixel, or bloom radius
	if (line.substr(0,5) == "width"){
	    read_num = sscanf(line.c_str(), "width %d", &h_width);
	    check_read(1, read_num, line_num);
	}
	else if (line.substr(0,6) == "height"){
	    read_num = sscanf(line.c_str(), "height %d", &h_height);
	    check_read(1, read_num, line_num);
	}
	else if (line.substr(0,7) == "samples"){
	    read_num = sscanf(line.c_str(), "samples %d", &h_samples);
	    check_read(1, read_num, line_num);
	}
	else if (line.substr(0,6) == "radius"){
	    read_num = sscanf(line.c_str(), "radius %d", &bloom_rad);
	    check_read(1, read_num, line_num);
	}
	else if (line.substr(0,8) == "constant"){ //constant texture
	    float cr, cg, cb, sr, sg, sb, d;
	    read_num = sscanf(line.c_str(), "constant {%f, %f, %f} {%f, %f, %f} %f",
			      &cr, &cg, &cb, &sr, &sg, &sb, &d);
	    check_read(7, read_num, line_num);
	    h_const.push_back({{cr,cg,cb},{sr,sg,sb}, d});
	}
	else if (line.substr(0,9) == "checkered"){ //checkered texture
	    float r1, g1, b1, r2, g2, b2, sr, sg, sb, sx, sy;
	    read_num = sscanf(line.c_str(), "checkered {%f, %f, %f} {%f, %f, %f} {%f, %f, %f} %f %f",
			      &r1, &g1, &b1, &r2, &g2, &b2, &sr, &sg, &sb, &sx, &sy);
	    check_read(11, read_num, line_num);
	    h_check.push_back({{r1,g1,b1},{r2,g2,b2},{sx,sy,1},{sr,sg,sb}});
	}
	else if (line.substr(0,3) == "pix"){ //image texture
	    char filename [100];
	    read_num = sscanf(line.c_str(), "pix %s", &filename);
	    check_read(1, read_num, line_num);
	    create_texture(std::string(filename));
	}
	else if (line.substr(0,3) == "box"){ //box geometry
	    float minx, miny, minz, maxx, maxy, maxz;
	    char scat[20];
	    char text[20];
	    int num;
	    read_num = sscanf(line.c_str(), "box {%f, %f, %f} {%f, %f, %f} %s %s %d",
			      &minx, &miny, &minz, &maxx, &maxy, &maxz, &scat, &text, &num);
	    check_read(9, read_num, line_num);
	    scatter_type s;
	    texture_type t;
	    get_scat(scat, text, s, t, line_num);
	    h_boxes.push_back({{maxx, maxy, maxz}, {minx, miny, minz}, s, t, num});
	}
	else if (line.substr(0,6) == "sphere"){
	    float x, y, z, r;
	    char scat[20];
	    char text[20];
	    int num;
	    read_num = sscanf(line.c_str(), "sphere {%f, %f, %f} %f %s %s %d",
			      &x, &y, &z, &r, &scat, &text, &num);
	    check_read(7, read_num, line_num);
	    scatter_type s;
	    texture_type t;
	    get_scat(scat, text, s, t, line_num);
	    h_spheres.push_back({{x,y,z}, r, s, t, num});
	}
	else if (line.substr(0,4) == "mesh"){ //triangle mesh
	    char filename[100];
	    float scale, tx, ty, tz;
	    char scat[20];
	    char text[20];
	    int num;
	    read_num = sscanf(line.c_str(), "mesh %s %f {%f, %f, %f} %s %s %d",
			      &filename, &scale, &tx, &ty, &tz, &scat, &text, &num);
	    check_read(8, read_num, line_num);
	    scatter_type s;
	    texture_type t;
	    get_scat(scat, text, s, t, line_num);
	    create_mesh(std::string(filename), scale, {tx, ty, tz}, s, t, num, meshes);
	    meshes++;
	}
    }
    if (!h_width || !h_height || !h_samples || !bloom_rad)
	print_error("Invalid quality parameters");
}
