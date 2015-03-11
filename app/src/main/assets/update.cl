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

__kernel void update_c(__global float *vertices,
                       write_only image2d_t output,
                       float dh,
                       float dt,
                       float c,
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

  __local float local_z[GROUP_SIZE][GROUP_SIZE];//(3+8+3)*14
  __local float local_nx[GROUP_SIZE][GROUP_SIZE];
  __local float local_ny[GROUP_SIZE][GROUP_SIZE];
  __local float local_nz[GROUP_SIZE][GROUP_SIZE];

  int jj = get_local_id(0);
  int ii = get_local_id(1);

  //load to local
  int image_index = i * image_size_x + j;

  local_z[ii][jj] = v[image_index].z;
  local_nx[ii][jj] = v[image_index].nx;
  local_ny[ii][jj] = v[image_index].ny;
  local_nz[ii][jj] = v[image_index].nz;

  barrier(CLK_LOCAL_MEM_FENCE);

  const float n = 0.75f;
  float3 light_direction = normalize((float3)(0.0f,0.2f,1.0f));

  int index, x, y, size_r = 0;
  float3 normal, position_l, position_h, light_fraction, l;
  float c1, c2, intensity = 0.0f, light_color = 1.0f;

  position_l = (float3)((float)(j * dh), (float)(i * dh), 0.0f);

  for(y = (int)i - SIZE; y <= (int)i + SIZE; ++y){
    for(x = (int)j - SIZE; x <= (int)j + SIZE; ++x){

      if(y < 0) {
        continue;
      }
      if(x < 0){
        continue;
      }
      if(y >= image_size_y){
        continue;
      }
      if(x >= image_size_x){
        continue;
      }

      if(x - j >=0 && x - j < GROUP_SIZE && y - i>=0 && y - i< GROUP_SIZE){
        position_h = (float3)((float)dh * x, (float)dh * y, local_z[ii][jj]);
        normal = (float3)(local_nx[ii][jj], local_ny[ii][jj], local_nz[ii][jj]);
      }else{
        index = y * image_size_x + x;
        position_h = (float3)(v[index].x, v[index].y, v[index].z);
        normal = (float3)(v[index].nx, v[index].ny, v[index].nz);
      }
      //normal = (float3)(v[index].nx, v[index].ny, v[index].nz);

      l = normalize(position_l - position_h);

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

  // kernel 2
  float D = 17500.0f * dh;
  float u01, u21, u11, u10, u12;
  u11 = local_z[ii][jj];
  if(ii != 0 && jj != 0 && ii != GROUP_SIZE - 1 && jj != GROUP_SIZE - 1){
    u01 = local_z[ii - 1][jj];
    u10 = local_z[ii][jj - 1];
    if(i == image_size_y - 1){
      u21 = local_z[ii][jj];
    }else{
      u21 = local_z[ii + 1][jj];
    }
    if(j == image_size_x - 1){
      u12 = local_z[ii][jj];
    }else{
      u12 = local_z[ii][jj + 1];
    }
  }else{
    if(i == 0){
      u01 = local_z[ii][jj];
    }else{
      u01 = v[image_index - image_size_x].z;
    }
    if(j == 0){
      u10 = local_z[ii][jj];
    }else{
      u10 = v[image_index - 1].z;
    }
    if(i == image_size_y - 1){
      u21 = local_z[ii][jj];
    }else{
      u21 = v[image_index + image_size_x].z;
    }
    if(j == image_size_x - 1){
      u12 = local_z[ii][jj];
    }else{
      u12 = v[image_index + 1].z;
    }
  }

  float p = (u01 + u21 - 4.0f*u11  + u10 + u12) / (4.0f * dh * dh);

  float f = c * c * (p - v[image_index].v / D);

  v[image_index].v += f * dt;

  float3 nx = (float3)((float)dh*2, 0.0f, u21 - u01);
  float3 ny = (float3)(0.0f, (float)dh*2, u12 - u10);
  float3 nn = normalize(cross(nx, ny));

  v[image_index].nx = nn.x;
  v[image_index].ny = nn.y;
  v[image_index].nz = nn.z;
}

#define drop_size 5

__kernel void update_u(
                       const __global float *vertices,
                       float dh,
                       float dt,
                       int random,
                       int image_size_x,
                       int image_size_y,
                       __global float *debug)
{
  float2 rain_point;

  int j = get_global_id(0);
  int i = get_global_id(1);

  if(j >= image_size_x || i >= image_size_y) return;

  int index = i * image_size_x + j;
  __global vertex * v = (__global vertex *) vertices;
  vertex vp = v[index];


  vp.z += vp.v * dt;

  int x, y;
  int index2;

  y = random / image_size_x;
  x = random % image_size_x;

  if(y >= image_size_y){
    v[index].z = vp.z;
    return;
  };

  index2 = y * image_size_x + x;
  rain_point = (float2)((float)(dh * x), (float)(dh * y));

  float2 cp = (float2)(vp.x, vp.y);
  float r;

  r = distance(rain_point, cp);
  if(r < drop_size * dh){
    if(random % 2){
      vp.z += 0.7f * r * cos(r / drop_size / dh * 1.57f);
    }else{
      vp.z -= 0.7f * r * cos(r / drop_size / dh * 1.57f);
    }
  }

  v[index].z = vp.z;
}