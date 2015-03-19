typedef struct {
  float x;
  float y;
  float z;
  float v;
  float nx;
  float ny;
  float nz;
} vertex;

#define SIZE 3

#define GROUP_SIZE 8

#define LOCAL_SIZE (GROUP_SIZE + SIZE * 2)

#define GROUP_AMOUNT (GROUP_SIZE * GROUP_SIZE)
#define LOCAL_AMOUNT (LOCAL_SIZE * LOCAL_SIZE)

#define ROW_OFFSET GROUP_AMOUNT / LOCAL_SIZE
#define COL_OFFSET GROUP_AMOUNT % LOCAL_SIZE

#define LOOP_COUNT (LOCAL_AMOUNT - 1) / GROUP_AMOUNT

__kernel void update_c(__global float *vertices,
                       write_only image2d_t output,
                       float dh,
                       int image_size_x,
                       int image_size_y,
                       __global float *debug)
{
  int j = get_global_id(0);
  int i = get_global_id(1);

  if(j >= image_size_x || i >= image_size_y) {
    barrier(CLK_LOCAL_MEM_FENCE);
    return;
  }

  __global vertex * v = (__global vertex *) vertices;

  __local vertex local_vertices[LOCAL_AMOUNT];//(3+8+3)*14

  int local_offset = get_local_id(0) + get_local_id(1) * GROUP_SIZE;
//  int global_offset = (local_offset / LOCAL_SIZE) * image_size_x + (local_offset % LOCAL_SIZE);

  int global_offset_x = local_offset % LOCAL_SIZE;
  int global_offset_y = local_offset / LOCAL_SIZE;
  int global_offset = global_offset_y * image_size_x + global_offset_x;

//  int start_global = (get_group_id(1) * GROUP_SIZE - SIZE) * image_size_x + (get_group_id(0) * GROUP_SIZE -SIZE);
//  int start_global = (get_global_id(1) - get_local_id(1) - SIZE) * image_size_x + get_global_id(0) - get_local_id(0) - SIZE;

  int start_global_x = get_global_id(0) - get_local_id(0) - SIZE;
  int start_global_y = get_global_id(1) - get_local_id(1) - SIZE;
  int start_global = start_global_y * image_size_x + start_global_x;

  int reserve_x = start_global_x + LOCAL_SIZE;

  int start_local = 0;
  int current_column = 0;


  // start_global -= ROW_OFFSET * image_size_x + COL_OFFSET;
  // start_local -= GROUP_AMOUNT;

  if (start_global_x+global_offset_x >= 0 &&
      start_global_x+global_offset_x < image_size_x &&
      start_global_y+global_offset_y >= 0 &&
      start_global_y+global_offset_y < image_size_y)
  {
    local_vertices[start_local + local_offset] = v[start_global + global_offset];
//  if(start_global + global_offset < image_size_x * image_size_y && start_global + global_offset >= 0){
//    local_vertices[start_local + local_offset] = v[start_global + global_offset];
//    local_vertices[final_local] = v[final_global];
  }else{
    //local_vertices[start_local + local_offset].v = -2.0f;
  }


  int k;
  for(k = 0; k < LOOP_COUNT; ++k){

    start_global_x += COL_OFFSET;
    start_global_y += ROW_OFFSET;
    start_global += ROW_OFFSET * image_size_x + COL_OFFSET;
    start_local += GROUP_AMOUNT;
    current_column += COL_OFFSET;
    if (current_column > LOCAL_SIZE)
    {
      current_column -= LOCAL_SIZE;
      start_global = start_global - LOCAL_SIZE + image_size_x;
      start_global_x -= LOCAL_SIZE;
      start_global_y ++;
    }



    int final_global = start_global + global_offset;
    int final_global_x = start_global_x + global_offset_x;
    int final_global_y = start_global_y + global_offset_y;

    int final_local = start_local + local_offset;
    if(final_local >= LOCAL_AMOUNT){
      break;
    }



    //current_column += (local_offset % LOCAL_SIZE);
    if (current_column + global_offset_x > LOCAL_SIZE)
    {
      final_global = final_global - LOCAL_SIZE + image_size_x;
      final_global_x -= LOCAL_SIZE;
      final_global_y ++;
    }

    if (final_global_y < 0 ||
        final_global_x >= reserve_x ||
        final_global_x < 0)
    {
      //local_vertices[final_local].v = -2.0f;
      continue;
    }


    if (final_global_x >=0 &&
        final_global_x < image_size_x &&
        final_global_y >=0 &&
        final_global_y < image_size_y){
    //if(final_global < image_size_x * image_size_y && final_global >= 0){
      //local_vertices[start_local + local_offset] = v[start_global + global_offset];
      local_vertices[final_local] = v[final_global];
    }else{
      //local_vertices[final_local].v = -1.0f;
    }
  }

  barrier(CLK_LOCAL_MEM_FENCE);
  //load to local

  if (i==3 && j==3){
  	int o=0;
    int p,q;
  	for(p = 0 - SIZE; p < 0 + GROUP_SIZE + SIZE; ++p){
      for(q = 0 - SIZE; q < 0 + GROUP_SIZE + SIZE; ++q){
        if(p >= 0 && q >=0){
          debug[o]=v[q * image_size_x + p].v;
        }
        o+=1;
      }
  	}
  	for (p=0; p<LOCAL_SIZE; p++){
      for (q=0; q<LOCAL_SIZE; q++){
        debug[o]=local_vertices[q * LOCAL_SIZE + p].v;
        o+=1;
      }
    }
  }

  const float n = 0.75f;
  float3 light_direction = normalize((float3)(0.0f,0.2f,1.0f));

  int index, x, y, size_r = 0;
  float3 normal, position_l, position_h, light_fraction, l;
  float c1, c2, intensity = 0.0f, light_color = 1.0f;

  position_l = (float3)((float)(j * dh), (float)(i * dh), 0.0f);

  //for(x = (int)j - SIZE; x <= (int)j + SIZE; ++x){
  //for(y = (int)i - SIZE; y <= (int)i + SIZE; ++y){
  for (x=(int)get_local_id(0); x<=(int)get_local_id(0)+SIZE*2; x++){
    for (y=(int)get_local_id(1); y<=(int)get_local_id(1)+SIZE*2; y++){
      /*if(y < 0) {
       continue;
       }
       if(x < 0){
       continue;
       }
       //if(y >= image_size_y){
       if (y > LOCAL_MEM_SIZE){
       continue;
       }
       //if(x >= image_size_x){
       if (x > LOCAL_MEM_SIZE){
       continue;
       }*/
      //index = y * image_size_x + x;
      index = y * LOCAL_SIZE + x;
      //position_h = (float3)(v[index].x, v[index].y, v[index].z);
      position_h = (float3)(local_vertices[index].x, local_vertices[index].y, local_vertices[index].z);

//      if (x==3 && y==3)
//      {
//        debug[0]=v[j * image_size_x + i].z;
//        debug[1]=position_h.z;
//        debug[2]=i;
//        debug[3]=j;
//      }

      l = normalize(position_l - position_h);

      //normal = normalize((float3)(v[index].nx, v[index].ny, v[index].nz));
      normal = normalize((float3)(local_vertices[index].nx, local_vertices[index].ny, local_vertices[index].nz));
      c1 = max(dot(normal, -l), 0.0f);
      light_fraction = -l/n - normal * (sqrt(1.0f - (1.0f - c1*c1)/n/n) - c1/n);
      c2 = max(dot(light_fraction, light_direction),0.0f);

      intensity += c2;

      normal = (float3)(0.0f, 0.0f, 1.0f);
      c1 = max(dot(normal, -l), 0.0f);
      light_fraction = -l/n - normal * (sqrt(1.0f - (1.0f - c1*c1)/n/n) - c1/n);
      c2 = max(dot(light_fraction, light_direction),0.0f);

      intensity -= c2;

      size_r ++;
    }
  }
  intensity = intensity / size_r;
  intensity = intensity + 0.2f;

  intensity = max(min(intensity, 0.5f), 0.0f) + 0.4f;

  write_imagef(output, (int2)(j,i), (float4)(intensity));
}


__kernel void update_v(
                       __global float *vertices,
                       float dh, float dt, float c,
                       uint image_size_x,
                       uint image_size_y,
                       __global float *debug)
{
  float D = 17500.0f * dh;
  uint j = get_global_id(0);
  uint i = get_global_id(1);

  if(j >= image_size_x || i >= image_size_y) return;

  uint xMax = image_size_x;
  uint yMax = image_size_y;
  uint xMax1 = xMax - 1;
  uint yMax1 = yMax - 1;
  uint index = i * xMax + j;
  __global vertex * v = (__global vertex *) vertices;

  float u00, u01, u02, u10, u11, u12, u20, u21, u22,
  x00, x01, x02, x10, x11, x12, x20, x21, x22,
  y00, y01, y02, y10, y11, y12, y20, y21, y22;

  u11 = v[index].z;
  x11 = v[index].x;
  y11 = v[index].y;

  if(j > 0 && i > 0 && j < xMax1 && i < yMax1){
    u00 = v[index -xMax -1].z;
    x00 = v[index -xMax -1].x;
    y00 = v[index -xMax -1].y;

    u01 = v[index -1].z;
    x01 = v[index -1].x;
    y01 = v[index -1].y;

    u02 = v[index +xMax -1].z;
    x02 = v[index +xMax -1].x;
    y02 = v[index +xMax -1].y;

    u10 = v[index -xMax].z;
    x10 = v[index -xMax].x;
    y10 = v[index -xMax].y;

    u12 = v[index +xMax].z;
    x12 = v[index +xMax].x;
    y12 = v[index +xMax].y;

    u20 = v[index -xMax +1].z;
    x20 = v[index -xMax +1].x;
    y20 = v[index -xMax +1].y;

    u21 = v[index +1].z;
    x21 = v[index +1].x;
    y21 = v[index +1].y;

    u22 = v[index +xMax +1].z;
    x22 = v[index +xMax +1].x;
    y22 = v[index +xMax +1].y;
  }else{
    if(j == 0 && i == 0){
      u00 = u11;
      x00 = x11;
      y00 = y11;

      u01 = u11;
      x01 = x11;
      y01 = y11;

      u02 = v[index +xMax].z;
      x02 = v[index +xMax].x;
      y02 = v[index +xMax].y;

      u10 = u11;
      x10 = x11;
      y10 = y11;

      u12 = u02;
      x12 = x02;
      y12 = y02;

      u20 = v[index +1].z;
      x20 = v[index +1].x;
      y20 = v[index +1].y;

      u21 = u20;
      x21 = x20;
      y21 = y20;

      u22 = v[index +xMax +1].z;
      x22 = v[index +xMax +1].x;
      y22 = v[index +xMax +1].y;
    }else if(j == 0 && i < yMax1){
      u00 = v[index -xMax].z;
      x00 = v[index -xMax].x;
      y00 = v[index -xMax].y;

      u01 = u11;
      x01 = x11;
      y01 = y11;

      u02 = v[index +xMax].z;
      x02 = v[index +xMax].x;
      y02 = v[index +xMax].y;

      u10 = u00;
      x10 = x00;
      y10 = y00;

      u12 = u02;
      x12 = x02;
      y12 = y02;

      u20 = v[index -xMax +1].z;
      x20 = v[index -xMax +1].x;
      y20 = v[index -xMax +1].y;

      u21 = v[index +1].z;
      x21 = v[index +1].x;
      y21 = v[index +1].y;

      u22 = v[index +xMax +1].z;
      x22 = v[index +xMax +1].x;
      y22 = v[index +xMax +1].y;
    }else if(j == 0 && i == yMax1){
      // u00 = u10
      u00 = v[index -xMax].z;
      x00 = v[index -xMax].x;
      y00 = v[index -xMax].y;

      u01 = u11;
      x01 = x11;
      y01 = y11;

      u02 = u11;
      x02 = x11;
      y02 = y11;

      u10 = u00;
      x10 = x00;
      y10 = y00;

      u12 = u11;
      x12 = x11;
      y12 = y11;

      u20 = v[index -xMax +1].z;
      x20 = v[index -xMax +1].x;
      y20 = v[index -xMax +1].y;

      u21 = v[index +1].z;
      x21 = v[index +1].x;
      y21 = v[index +1].y;

      u22 = u21;
      x22 = x21;
      y22 = y21;
    }else if(i == yMax1 && j < xMax1){
      u00 = v[index -xMax -1].z;
      x00 = v[index -xMax -1].x;
      y00 = v[index -xMax -1].y;

      u01 = v[index -1].z;
      x01 = v[index -1].x;
      y01 = v[index -1].y;

      u02 = u01;
      x02 = x01;
      y02 = y01;

      u10 = v[index -xMax].z;
      x10 = v[index -xMax].x;
      y10 = v[index -xMax].y;

      u12 = u11;
      x12 = x11;
      y12 = y11;

      u20 = v[index -xMax +1].z;
      x20 = v[index -xMax +1].x;
      y20 = v[index -xMax +1].y;

      u21 = v[index +1].z;
      x21 = v[index +1].x;
      y21 = v[index +1].y;

      u22 = u21;
      x22 = x21;
      y22 = y21;
    }else if(i == yMax1 && j == xMax1){
      u00 = v[index -xMax -1].z;
      x00 = v[index -xMax -1].x;
      y00 = v[index -xMax -1].y;

      u01 = v[index -1].z;
      x01 = v[index -1].x;
      y01 = v[index -1].y;

      u02 = u01;
      x02 = x01;
      y02 = y01;

      u10 = v[index -xMax].z;
      x10 = v[index -xMax].x;
      y10 = v[index -xMax].y;

      u12 = u11;
      x12 = x11;
      y12 = y11;

      u20 = u10;
      x20 = x10;
      y20 = y10;

      u21 = u11;
      x21 = x11;
      y21 = y11;

      u22 = u11;
      x22 = x11;
      y22 = y11;
    }else if(j == xMax1 && i > 0){
      u00 = v[index -xMax -1].z;
      x00 = v[index -xMax -1].x;
      y00 = v[index -xMax -1].y;

      u01 = v[index -1].z;
      x01 = v[index -1].x;
      y01 = v[index -1].y;

      u02 = v[index +xMax -1].z;
      x02 = v[index +xMax -1].x;
      y02 = v[index +xMax -1].y;

      u10 = v[index -xMax].z;
      x10 = v[index -xMax].x;
      y10 = v[index -xMax].y;

      u12 = v[index +xMax].z;
      x12 = v[index +xMax].x;
      y12 = v[index +xMax].y;

      u20 = u10;
      x20 = x10;
      y20 = y10;

      u21 = u11;
      x21 = x11;
      y21 = y11;

      u22 = u12;
      x22 = x12;
      y22 = y12;
    }else if(i == 0 && j== xMax1){
      u00 = v[index -1].z;
      x00 = v[index -1].x;
      y00 = v[index -1].y;

      u01 = u00;
      x01 = x00;
      y01 = y00;

      u02 = v[index +xMax -1].z;
      x02 = v[index +xMax -1].x;
      y02 = v[index +xMax -1].y;

      u10 = u11;
      x10 = x11;
      y10 = y11;

      u12 = v[index +xMax].z;
      x12 = v[index +xMax].x;
      y12 = v[index +xMax].y;

      u20 = u11;
      x20 = x11;
      y20 = y11;

      u21 = u11;
      x21 = x11;
      y21 = y11;

      u22 = u12;
      x22 = x12;
      y22 = y12;
    }else{
      u00 = v[index -1].z;
      x00 = v[index -1].x;
      y00 = v[index -1].y;

      u01 = u00;
      x01 = x00;
      y01 = y00;

      u02 = v[index +xMax -1].z;
      x02 = v[index +xMax -1].x;
      y02 = v[index +xMax -1].y;

      u10 = u11;
      x10 = x11;
      y10 = y11;

      u12 = v[index +xMax].z;
      x12 = v[index +xMax].x;
      y12 = v[index +xMax].y;

      u20 = v[index +1].z;
      x20 = v[index +1].x;
      y20 = v[index +1].y;

      u21 = u20;
      x21 = x20;
      y21 = y20;

      u22 = v[index +xMax +1].z;
      x22 = v[index +xMax +1].x;
      y22 = v[index +xMax +1].y;
    }
  }
  float p = (u00 + 4.0f*u01 + 4.0f*u21 + u22 - 20.0f*u11 + u20 + 4.0f*u10 + 4.0f*u12 + u02) / (6.0f * dh * dh);
  //float p = (u01 + u21 - 4.0f*u11  + u10 + u12) / (4.0f * dh * dh);

  float f = c * c * (p - v[index].v / D);

  v[index].v += f * dt;

  float3 nx = (float3)(2.0f*(x21 - x01) + x20 - x00 + x22 - x02, 0.0f, 2.0f*(u21 - u01) + u20 - u00 + u22 - u02);
  float3 ny = (float3)(0, 2.0f*(y12 - y10) + y02 - y00 + y22 - y20, 2.0f * (u12 - u10) + u02 - u00 + u22 - u20);
  //float3 nx = (float3)(x21 - x01, 0.0f, u21 - u01);
  //float3 ny = (float3)(0.0f, y12 - y10, u12 - u10);
  float3 n = normalize(cross(nx, ny));

  v[index].nx = n.x;
  v[index].ny = n.y;
  v[index].nz = n.z;
}

#define drop_size 5

__kernel void update_u(
                       __global float *vertices,
                       float dh,
                       float dt,
                       int random,
                       int xMax,
                       int yMax,
                       __global float *debug)
{
  float2 rain_point;

  int j = get_global_id(0);
  int i = get_global_id(1);

  if(j >= xMax || i >= yMax) return;

  int index = i * xMax + j;
  __global vertex * v = (__global vertex *) vertices;
  vertex vp = v[index];

  vp.z += vp.v * dt;

  int x, y;
  int index2;

  y = random / xMax;
  x = random % yMax;

  if(y >= yMax){
    v[index].z = vp.z;
    return;
  };

  index2 = y * xMax + x;
  rain_point = (float2)(v[index2].x, v[index2].y);

  float2 cp = (float2)(vp.x, vp.y);
  float r;

  r = distance(rain_point, cp);
  if(r < drop_size * dh){
    if(random % 2){
      vp.z += 0.2f * r * cos(r / drop_size / dh * 1.57f);
    }else{
      vp.z -= 0.2f * r * cos(r / drop_size / dh * 1.57f);
    }
  }

  v[index].z = vp.z;
}