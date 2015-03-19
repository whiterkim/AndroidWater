//
//  cl_host.cpp
//  Rain
//
//  Created by hyspace on 3/4/15.
//  Copyright (c) 2015 hyspace. All rights reserved.
//

#include "cl_host.h"
#include "heightfield.h"
#define DEBUG_SIZE 10

cl::Context context;
cl::CommandQueue queue;
cl::Kernel update_caustic;
cl::Kernel update_v;
cl::Kernel update_u;


std::vector<cl::Memory> cgl_objs;
cl::Buffer debug_buff;
GLfloat debug_buffh[DEBUG_SIZE];

int initCL(std::string& prog, EGLContext& kEGLContext, EGLDisplay& kEGLDisplay)
{
  cl_int err;

  // get platform
  std::vector<cl::Platform> platformList;
  cl::Platform::get(&platformList);
  std::string platformVendor;
  platformList[0].getInfo((cl_platform_info)CL_PLATFORM_VENDOR, &platformVendor);

  // create OpenCL context with OpenGL context

  cl_context_properties props[] =
  {
    CL_GL_CONTEXT_KHR, (cl_context_properties) kEGLContext,
    CL_EGL_DISPLAY_KHR, (cl_context_properties) kEGLDisplay,
    CL_CONTEXT_PLATFORM, (cl_context_properties)(platformList[0])(),
    0
  };

  context = cl::Context(
                        CL_DEVICE_TYPE_GPU,
                        props,
                        NULL,
                        NULL,
                        &err);

  if (err != CL_SUCCESS){
    LOGW("Error: Failed to create context\n");
    return EXIT_FAILURE;
  }

  // create command queue
  std::vector<cl::Device> devices;
  devices = context.getInfo<CL_CONTEXT_DEVICES>();

  queue = cl::CommandQueue(context, devices[0], 0, &err);
  if (err != CL_SUCCESS)
  {
    LOGW("Error: Failed to create a cl queue!\n");
    return EXIT_FAILURE;
  }

  // compile OpenCL program.
  cl::Program::Sources source(1,
                              std::make_pair(prog.c_str(), prog.length()+1)
                              );
  cl::Program program = cl::Program(context, source);
  err = program.build(devices, "-cl-single-precision-constant -cl-strict-aliasing -cl-fast-relaxed-math");
  if (err != CL_SUCCESS)
  {
    LOGI("Error: Failed to build\n");
    std::string errinfo;
    err = program.getBuildInfo(devices[0], CL_PROGRAM_BUILD_LOG, &errinfo);
    LOGW("%s\n", errinfo.c_str());
    return EXIT_FAILURE;
  }

  // build OpenCL kernels.
  update_caustic = cl::Kernel(program, "update_c", &err);
  if (err != CL_SUCCESS)
  {
    LOGI("Failed to create kernel \"update_caustic\"!\n");
    return EXIT_FAILURE;
  }

//  update_v = cl::Kernel(program, "update_v", &err);
//  if (err != CL_SUCCESS)
//  {
//    LOGI("Failed to create kernel \"update_v\"!\n");
//    return EXIT_FAILURE;
//  }

  update_u = cl::Kernel(program, "update_u", &err);
  if (err != CL_SUCCESS)
  {
    LOGI("Failed to create kernel \"update_u\"!\n");
    return EXIT_FAILURE;
  }

  // create OpenCL buffer for debug
  debug_buff = cl::Buffer(context,
                          CL_MEM_READ_WRITE,
                          sizeof(GLfloat) * DEBUG_SIZE,
                          NULL,
                          &err);
  if (err != CL_SUCCESS)
  {
    LOGI("Failed to create debug buffer!\n");
    return EXIT_FAILURE;
  }

  // set random seed
  srand((unsigned int) time(NULL));

  return CL_SUCCESS;
}

int initGLObjects(GLuint vertex_buffer_id, GLuint texture_id){
  cl_int err;
  cl::Buffer cl_vertex_buffer;
  cl::Image2D caustic_texture;

  // get buffer object ptr from OpenGL
  cl_vertex_buffer = cl::BufferGL(context,
                                  CL_MEM_READ_WRITE,
                                  vertex_buffer_id,
                                  &err);
  if (err != CL_SUCCESS)
  {
    LOGI("Failed to create vertex_buffer!\n");
    return EXIT_FAILURE;
  }

  // get image object ptr from OpenGL
  caustic_texture = cl::Image2DGL(
                                   context,
                                   CL_MEM_READ_WRITE,
                                   GL_TEXTURE_2D,
                                   0,
                                   texture_id,
                                   &err);
  if (err != CL_SUCCESS)
  {
    LOGI("Failed to create texture_buffer!\n");
    return EXIT_FAILURE;
  }

  // put OpenGL objects to list
  cgl_objs.push_back(cl_vertex_buffer);
  cgl_objs.push_back(caustic_texture);

  return CL_SUCCESS;
}

int freeCL(){
  return CL_SUCCESS;
}

int recompute(GLuint xMax, GLuint yMax, GLfloat dh, GLfloat dt, GLfloat c)
{
  const int ls = 8;
  uint localx = xMax + (xMax % ls ? ls - xMax % ls : 0);
  uint localy = yMax + (yMax % ls ? ls - yMax % ls : 0);

  cl::NDRange offset = cl::NullRange;
  cl::NDRange global = cl::NDRange((size_t)localx, (size_t)localy);
  cl::NDRange local = cl::NDRange((size_t)ls, (size_t)ls);

  int r = rand() % 10;
  if(r == 0){
    r = rand() % (xMax * yMax);
  }else{
    r = xMax * yMax + 1;
  }
//  r = xMax * 50 +50;

  cl_int err;

  // acquire GL objects
  err = queue.enqueueAcquireGLObjects(&cgl_objs, NULL, NULL);
  if (err != CL_SUCCESS)
  {
    LOGI("Failed to acquire GL objects \n");
    return EXIT_FAILURE;
  }
  queue.finish();


//  // launch kernels
//  update_v.setArg(0, cgl_objs[0]);
//  update_v.setArg(1, dh);
//  update_v.setArg(2, dt);
//  update_v.setArg(3, c);
//  update_v.setArg(4, xMax);
//  update_v.setArg(5, yMax);
//  update_v.setArg(6, debug_buff);
//
//  err = queue.enqueueNDRangeKernel(update_v,
//                                  offset,
//                                  global,
//                                  local,
//                                  NULL,
//                                  NULL);
//  if (err != CL_SUCCESS)
//  {
//   LOGI("Failed to run kernel \"update_v\" %d \n", err);
//   return EXIT_FAILURE;
//  }
//  queue.finish();

  update_u.setArg(0, cgl_objs[0]);
  update_u.setArg(1, dh);
  update_u.setArg(2, dt);
  update_u.setArg(3, r);
  update_u.setArg(4, xMax);
  update_u.setArg(5, yMax);
  update_u.setArg(6, debug_buff);

  err = queue.enqueueNDRangeKernel(update_u,
                                   offset,
                                   global,
                                   local,
                                   NULL,
                                   NULL);
  if (err != CL_SUCCESS)
  {
    LOGI("Failed to run kernel \"update_u\" %d\n", err);
    return EXIT_FAILURE;
  }
  queue.finish();

  update_caustic.setArg(0, cgl_objs[0]);
  update_caustic.setArg(1, cgl_objs[1]);
  update_caustic.setArg(2, dh);
  update_caustic.setArg(3, dt);
  update_caustic.setArg(4, c);
  update_caustic.setArg(5, xMax);
  update_caustic.setArg(6, yMax);
  update_caustic.setArg(7, debug_buff);
  err = queue.enqueueNDRangeKernel(update_caustic,
                                   cl::NullRange,
                                   global,
                                   local,
                                   NULL,
                                   NULL
                                   );
  if (err != CL_SUCCESS)
  {
    LOGI("Failed to run kernel \"update_caustic\" %d\n", err);
    return EXIT_FAILURE;
  }
  queue.finish();

//  // output debug info
//  err = queue.enqueueReadBuffer(debug_buff,
//                               CL_TRUE,
//                               0,
//                               sizeof(GLfloat) * DEBUG_SIZE,
//                               &debug_buffh);
//  for(int i = 0; i < DEBUG_SIZE; ++i){
//   LOGI("debug[%d]:%f\n", i, debug_buffh[i]);
//  }
//  LOGI("=========\n");
//  if (err != CL_SUCCESS)
//  {
//   LOGI("Failed to read %d \n", err);
//   return EXIT_FAILURE;
//  }

  // release GL objects
  err = queue.enqueueReleaseGLObjects(&cgl_objs, NULL, NULL);
  if (err != CL_SUCCESS)
  {
    LOGI("Failed to release GL objects \n");
    return EXIT_FAILURE;
  }
  queue.finish();
}

int recomputeCPU(GLuint xMax, GLuint yMax, GLfloat dh, GLfloat dt, GLfloat c, GLfloat* vertex_buffer_cpu, GLuint caustic_texture_id)
{
  const int ls = 8;
  uint localx = xMax + (xMax % ls ? ls - xMax % ls : 0);
  uint localy = yMax + (yMax % ls ? ls - yMax % ls : 0);

  int r = rand() % 10;
  if(r == 0){
    r = rand() % (xMax * yMax);
  }else{
    r = xMax * yMax + 1;
  }
//  r = xMax * 50 +50;

  cl_int err;


  //LOGI("K1 start");
  ///////////////////////////////////////////////////////////
  vertex *v = (vertex*) vertex_buffer_cpu;
  int random=r;
  int image_size_x=xMax;
  int image_size_y=yMax;
  int drop_size=5;
  ///////////////////////////////////////////////////////////
  for (int i=0; i< image_size_y; i++)
  {
     for (int j=0; j<image_size_x; j++)
        {
          glm::vec2 rain_point;

          int index = i * image_size_x + j;
          vertex vp = v[index];


          vp.z += vp.v * dt;

          int x, y;
          int index2;

          y = random / image_size_x;
          x = random % image_size_x;

          if(y >= image_size_y){
            v[index].z = vp.z;
            continue;//return EXIT_FAILURE;
          };

          index2 = y * image_size_x + x;
          rain_point = glm::vec2((float)(dh * x), (float)(dh * y));

          glm::vec2 cp(vp.x, vp.y);
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
    }
  //LOGI("K1 finish");
  ///////////////////////////////////////////////////////////
  int SIZE=3;
  int GROUP_SIZE=8;
  ///////////////////////////////////////////////////////////
  GLubyte intensity_array[xMax][yMax];
  for (int i=0; i<image_size_y; i++)
    for (int j=0; j<image_size_x; j++)
        {

          const float n = 0.75f;
          glm::vec3 light_direction(0.0f,0.5f,1.0f);
          light_direction = glm::normalize(light_direction);

          int index, x, y, size_r = 0;
          glm::vec3 normal, position_l, position_h, light_fraction, l;
          float c1, c2, intensity = 0.0f, light_color = 1.0f;

          position_l = glm::vec3((float)(j * dh), (float)(i * dh), 0.0f);

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

                index = y * image_size_x + x;
                position_h = glm::vec3(v[index].x, v[index].y, v[index].z);
                normal = glm::vec3(v[index].nx, v[index].ny, v[index].nz);
              //normal = (float3)(v[index].nx, v[index].ny, v[index].nz);

              l = normalize(position_l - position_h);

              c1 = max(dot(normal, -l), 0.0f);
              light_fraction = -l/n - normal * (sqrt(1.0f - (1.0f - c1*c1)/n/n) - c1/n);
              c2 = max(dot(light_fraction, light_direction),0.0f);

              intensity += c2;

              normal = glm::vec3(0.0f, 0.0f, 1.0f);
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

          //write_imagef(output, (int2)(j,i), (float4)(intensity));
          //glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, height_field->xMax, height_field->yMax, 0, GL_RED, GL_UNSIGNED_BYTE, NULL);
          //GLubyte temp=200;

          //glTexSubImage2D( GL_TEXTURE_2D,  0,  j,  i,  1,  1,  GL_RED,  GL_UNSIGNED_BYTE,  &temp);

          intensity_array[j][i]=(intensity*256);
        }
  //glEnable(GL_TEXTURE_2D);
  //glActiveTexture(GL_TEXTURE1);
  glBindTexture(GL_TEXTURE_2D, caustic_texture_id);
  //LOGI("intensity %f",intensity_array[1][1]);
  //glTexSubImage2D( GL_TEXTURE_2D,  0,  0,  0,  xMax,  yMax,  GL_RED,  GL_UNSIGNED_BYTE,  intensity_array);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, xMax, yMax, 0, GL_RED, GL_UNSIGNED_BYTE, intensity_array);
  //LOGI("K2 finish");
  ///////////////////////////////////////////////////////////
  for (int i=0; i<image_size_y; i++)
      for (int j=0; j<image_size_x; j++)
          {
            float D = 10500.0f * dh;

            uint xMax = image_size_x;
            uint yMax = image_size_y;
            uint xMax1 = xMax - 1;
            uint yMax1 = yMax - 1;
            uint index = i * xMax + j;

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

            glm::vec3 nx = glm::vec3(2.0f*(x21 - x01) + x20 - x00 + x22 - x02, 0.0f, 2.0f*(u21 - u01) + u20 - u00 + u22 - u02);
            glm::vec3 ny = glm::vec3(0, 2.0f*(y12 - y10) + y02 - y00 + y22 - y20, 2.0f * (u12 - u10) + u02 - u00 + u22 - u20);
            //float3 nx = (float3)(x21 - x01, 0.0f, u21 - u01);
            //float3 ny = (float3)(0.0f, y12 - y10, u12 - u10);
            glm::vec3 n = glm::cross(nx, ny);
            n = glm::normalize(n);

            v[index].nx = n.x;
            v[index].ny = n.y;
            v[index].nz = n.z;
            //v[index].nz = 100;
          }
  //LOGI("K3 finish");
  ///////////////////////////////////////////////////////////



}