//
//  cl_host.cpp
//  Rain
//
//  Created by hyspace on 3/4/15.
//  Copyright (c) 2015 hyspace. All rights reserved.
//

#include "cl_host.h"
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

  update_v = cl::Kernel(program, "update_v", &err);
  if (err != CL_SUCCESS)
  {
    LOGI("Failed to create kernel \"update_v\"!\n");
    return EXIT_FAILURE;
  }

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


  // launch kernels
  update_v.setArg(0, cgl_objs[0]);
  update_v.setArg(1, dh);
  update_v.setArg(2, dt);
  update_v.setArg(3, c);
  update_v.setArg(4, xMax);
  update_v.setArg(5, yMax);
  update_v.setArg(6, debug_buff);

  err = queue.enqueueNDRangeKernel(update_v,
                                  offset,
                                  global,
                                  local,
                                  NULL,
                                  NULL);
  if (err != CL_SUCCESS)
  {
   LOGI("Failed to run kernel \"update_v\" %d \n", err);
   return EXIT_FAILURE;
  }
  queue.finish();

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
  update_caustic.setArg(3, xMax);
  update_caustic.setArg(4, yMax);
  update_caustic.setArg(5, debug_buff);

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
//  for(int i = 0; i < 10; ++i){
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