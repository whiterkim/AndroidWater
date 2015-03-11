/*
 * Copyright (C) 2010 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

//BEGIN_INCLUDE(all)
#include <jni.h>
#include <errno.h>

#include <EGL/egl.h>
#include <GLES3/gl3.h>

#include <android_native_app_glue.h>

#include <iostream>
#include <cmath>

#include "heightfield.h"
#include "AssetLoader.h"
#include "cl_host.h"
#include "lodepng.h"
#include "log.h"

using namespace std;

/**
 * Our saved state data.
 */
struct saved_state {
    float angle;
    int32_t x;
    int32_t y;
};

/**
 * Shared state for our app.
 */
struct engine {
    struct android_app* app;
    int initialized = 0;
    int animating;
    EGLDisplay display;
    EGLSurface surface;
    EGLContext context;
    int32_t width;
    int32_t height;
    struct saved_state state;
};

HeightField<GLfloat> * height_field;

GLfloat dh = 16.0;
GLfloat c = 600.0;
const double dt = dh / c / 1.5;

GLuint vertex_buffer;
GLuint index_buffer;
GLuint vertex_array_id;
GLuint program_id;

GLuint caustic_texture_id;
GLuint sc_texture_id;

Resource *r;

double time_now, time_fps;
unsigned int frame_counter = 0;

typedef struct {
  glm::vec3 direction;
  glm::vec3 intensities;
}D_light;

/**
 * Initialize an EGL context for the current display.
 */
static int engine_init_display(struct engine* engine) {
    // initialize OpenGL ES and EGL

    /*
     * Here specify the attributes of the desired configuration.
     * Below, we select an EGLConfig with at least 8 bits per color
     * component compatible with on-screen windows
     */
    const EGLint attribs[] = {
            EGL_SURFACE_TYPE, EGL_WINDOW_BIT,
            EGL_BLUE_SIZE, 8,
            EGL_GREEN_SIZE, 8,
            EGL_RED_SIZE, 8,
            EGL_NONE
    };
    const EGLint ctxattr[] = {
     		EGL_CONTEXT_CLIENT_VERSION, 3,
    		EGL_NONE
    };

    EGLint w, h, dummy, format;
    EGLint numConfigs;
    EGLConfig config;
    EGLSurface surface;
    EGLContext context;

    EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);

    eglInitialize(display, 0, 0);

    /* Here, the application chooses the configuration it desires. In this
     * sample, we have a very simplified selection process, where we pick
     * the first EGLConfig that matches our criteria */
    eglChooseConfig(display, attribs, &config, 1, &numConfigs);

    /* EGL_NATIVE_VISUAL_ID is an attribute of the EGLConfig that is
     * guaranteed to be accepted by ANativeWindow_setBuffersGeometry().
     * As soon as we picked a EGLConfig, we can safely reconfigure the
     * ANativeWindow buffers to match, using EGL_NATIVE_VISUAL_ID. */
    eglGetConfigAttrib(display, config, EGL_NATIVE_VISUAL_ID, &format);

    ANativeWindow_setBuffersGeometry(engine->app->window, 0, 0, format);

    surface = eglCreateWindowSurface(display, config, engine->app->window, NULL);
    context = eglCreateContext(display, config, NULL, ctxattr);

    if (eglMakeCurrent(display, surface, surface, context) == EGL_FALSE) {
        LOGW("Unable to eglMakeCurrent");
        return -1;
    }

    eglQuerySurface(display, surface, EGL_WIDTH, &w);
    eglQuerySurface(display, surface, EGL_HEIGHT, &h);

    engine->display = display;
    engine->context = context;
    engine->surface = surface;

    //========================
    // GLES Setup - shaders
    //========================

    LOGI("Draw area size: %dx%d", w, h);

    vector<unsigned char> buffer2;
    r->read("update.cl", buffer2);
    string cl = string(buffer2.begin(), buffer2.end());
    initCL(cl, engine->context, engine->display);
    vector<unsigned char> buffer;
    r->read("vertex.glsl", buffer);
    string vs = string(buffer.begin(), buffer.end());
    r->read("fragment.glsl", buffer);
    string fs = string(buffer.begin(), buffer.end());

    GLuint vertexShader;
    GLuint fragmentShader;
    char const * s;
    int InfoLogLength, err = 0;

    vertexShader = glCreateShader(GL_VERTEX_SHADER);
    s = vs.c_str();
    glShaderSource(vertexShader, 1, &s, NULL);
    glCompileShader(vertexShader);
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &err);
    if(err == GL_FALSE){
      glGetShaderiv(vertexShader, GL_INFO_LOG_LENGTH, &InfoLogLength);
      buffer.resize(InfoLogLength);
      glGetShaderInfoLog(vertexShader, InfoLogLength, NULL, (GLchar *)&buffer[0]);
      LOGW("\nvs compile info:\n%s", &buffer[0]);
    }

    fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    s = fs.c_str();
    glShaderSource(fragmentShader, 1, &s, NULL);
    glCompileShader(fragmentShader);
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &err);
    if(err == GL_FALSE){
      glGetShaderiv(fragmentShader, GL_INFO_LOG_LENGTH, &InfoLogLength);
      buffer.resize(InfoLogLength);
      glGetShaderInfoLog(fragmentShader, InfoLogLength, NULL, (GLchar *)&buffer[0]);
      LOGW("fs compile error:%s", &buffer[0]);
    }

    program_id = glCreateProgram();
    glAttachShader(program_id, vertexShader);
    glAttachShader(program_id, fragmentShader);

    glLinkProgram(program_id);

    glUseProgram(program_id);

    //========================
    // GLES Setup - camera
    //========================

    GLfloat fovy = glm::radians(30.0f);
    GLfloat ratio = (GLfloat) w / (GLfloat) h;
    w = h; // make draw area square
    glm::mat4 projection = glm::perspective(fovy, ratio, 0.1f, 10000.0f);
    glm::vec3 camera_position = glm::vec3((GLfloat)w/2.0, (GLfloat)h/2.0, (GLfloat)h / ratio / 2.0 / tanf(fovy / 2.0));
    glm::mat4 view = glm::lookAt(camera_position, glm::vec3((GLfloat)w/2.0, (GLfloat)h/2.0, 0.0), glm::vec3(1.0,0.0,0.0));
    glm::mat4 model = glm::mat4(1.0f);
    glm::mat4 mvp = projection * view * model;
    GLuint camera_handler = glGetUniformLocation(program_id, "camera_position");
    glUniform3fv(camera_handler, 1, glm::value_ptr(camera_position));
    GLuint matrix_id = glGetUniformLocation(program_id, "MVP");
    glUniformMatrix4fv(matrix_id, 1, GL_FALSE, glm::value_ptr(mvp));

    //========================
    // GLES Setup - vertices
    //========================

    height_field = new HeightField<GLfloat> (w, h,(GLuint) dh);
    LOGI("Vertex buffer size: %dx%d", height_field->xMax, height_field->yMax);

    glGenVertexArrays(1, &vertex_array_id);
    glBindVertexArray(vertex_array_id);

    glGenBuffers(1, &vertex_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * height_field->vertex_length, height_field->vertex_buffer, GL_DYNAMIC_DRAW);

    glGenBuffers(1, &index_buffer);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint) * height_field->index_length, height_field->index_buffer, GL_DYNAMIC_DRAW);

    D_light light;
    light.direction = glm::normalize(glm::vec3(0.0,0.5,1.0));
    light.intensities = glm::vec3(0.98,0.9,0.6);
    GLuint light_direction_handler = glGetUniformLocation(program_id, "light.direction");
    glUniform3fv(light_direction_handler, 1, glm::value_ptr(light.direction));
    GLuint light_intensities_handler = glGetUniformLocation(program_id, "light.intensities");
    glUniform3fv(light_intensities_handler, 1, glm::value_ptr(light.intensities));

    GLuint vertex_handler = glGetAttribLocation(program_id, "vertex_position");
    glEnableVertexAttribArray(vertex_handler);
    glVertexAttribPointer(
                        vertex_handler,     // attribute. No partgcicular reason for 0, but must match the layout in the shader.
                        3,                  // size
                        GL_FLOAT,           // type
                        GL_FALSE,           // normalized?
                        sizeof(vertex), NULL);
    GLuint normal_handler = glGetAttribLocation(program_id, "vertex_normal");
    glEnableVertexAttribArray(normal_handler);
    glVertexAttribPointer(
                        normal_handler,     // attribute. No partgcicular reason for 0, but must match the layout in the shader.
                        3,                  // size
                        GL_FLOAT,           // type
                        GL_TRUE,            // normalized?
                        sizeof(vertex), (const GLvoid*)(4 * sizeof(GLfloat)));

    glm::vec2 size = glm::vec2(w, h);
    GLuint size_handler = glGetUniformLocation(program_id, "size");
    glUniform2fv(size_handler, 1, glm::value_ptr(size));

    //========================
    // GLES Setup - textures
    //========================

    std::vector<unsigned char> image;
    std::vector<unsigned char> bufferpng;
    int pngsize;
    pngsize = r->read("miku.png", bufferpng);

    unsigned iw, ih;
    unsigned error = lodepng::decode(image, iw, ih, bufferpng);
    if(error){
      LOGI("load png error");
    }

    glActiveTexture(GL_TEXTURE0);
    glGenTextures(1, &sc_texture_id);
    glBindTexture(GL_TEXTURE_2D, sc_texture_id);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, iw, ih,
                 0, GL_RGBA, GL_UNSIGNED_BYTE, &image[0]);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glGenerateMipmap(GL_TEXTURE_2D);
    GLuint sc_texture_handler = glGetUniformLocation(program_id, "sc_texture");
    glUniform1i(sc_texture_handler, 0);

    glActiveTexture(GL_TEXTURE1);
    glGenTextures(1, &caustic_texture_id);
    glBindTexture(GL_TEXTURE_2D, caustic_texture_id);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, height_field->xMax, height_field->yMax, 0, GL_RED, GL_UNSIGNED_BYTE, NULL);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glGenerateMipmap(GL_TEXTURE_2D);
    GLuint caustic_texture_handler = glGetUniformLocation(program_id, "caustic_texture");
    glUniform1i(caustic_texture_handler, 1);

    initGLObjects(vertex_buffer, caustic_texture_id);

    return 0;
}

/**
 * Just the current frame in the display.
 */
static void engine_draw_frame(struct engine* engine) {
    if (engine->display == NULL) {
        // No display.
        return;
    }

    glClear(GL_COLOR_BUFFER_BIT);

    // regenerate mipmap for caustic shader
    glActiveTexture(GL_TEXTURE1);
    glGenTextures(1, &caustic_texture_id);
    glGenerateMipmap(GL_TEXTURE_2D);

    // draw!
    glDrawElements(
                   GL_TRIANGLE_STRIP,  // mode
                   height_field->index_length,    // count
                   GL_UNSIGNED_INT,    // type
                   (void*)0            // element array buffer offset
                   );
    glFinish();
    eglSwapBuffers(engine->display, engine->surface);

    double t = now_s();
    ++frame_counter;
    if(t - time_fps > 1.0){
      double fps = (double)frame_counter / (t - time_fps);
      LOGI("fps: %f", fps);
      frame_counter = 0;
      time_fps = t;
    }


    // call OpenCL kernel
    int err = recompute(height_field->xMax, height_field->yMax, dh, min((GLfloat)(t - time_now), (GLfloat)dt), c);
    if(err != 0){
        engine->animating = 0;
    }

    time_now = now_s();
}

/**
 * Tear down the EGL context currently associated with the display.
 */
static void engine_term_display(struct engine* engine) {
    if (engine->display != EGL_NO_DISPLAY) {
        eglMakeCurrent(engine->display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        if (engine->context != EGL_NO_CONTEXT) {
            eglDestroyContext(engine->display, engine->context);
        }
        if (engine->surface != EGL_NO_SURFACE) {
            eglDestroySurface(engine->display, engine->surface);
        }
        eglTerminate(engine->display);
    }
    engine->animating = 0;
    engine->display = EGL_NO_DISPLAY;
    engine->context = EGL_NO_CONTEXT;
    engine->surface = EGL_NO_SURFACE;
}

/**
 * Process the next main command.
 */
static void engine_handle_cmd(struct android_app* app, int32_t cmd) {
    struct engine* engine = (struct engine*)app->userData;
    switch (cmd) {
        case APP_CMD_SAVE_STATE:
            // The system has asked us to save our current state.  Do so.
            engine->app->savedState = malloc(sizeof(struct saved_state));
            *((struct saved_state*)engine->app->savedState) = engine->state;
            engine->app->savedStateSize = sizeof(struct saved_state);

            LOGI("Status change: SAVE_STATE");
            break;
        case APP_CMD_INIT_WINDOW:
            // The window is being shown, get it ready.
            if (engine->app->window != NULL) {
                engine_init_display(engine);
                engine_draw_frame(engine);
            }

            LOGI("Status change: INIT_WINDOW");
            break;
        case APP_CMD_TERM_WINDOW:
            // The window is being hidden or closed, clean it up.
            engine_term_display(engine);

            LOGI("Status change: TERM_WINDOW");
            break;
        case APP_CMD_GAINED_FOCUS:
            engine->animating = 1;

            LOGI("Status change: GAINED_FOCUS");
            break;
        case APP_CMD_LOST_FOCUS:
            // Also stop animating.
            engine->animating = 0;
            engine_draw_frame(engine);

            LOGI("Status change: LOST_FOCUS");
            break;
    }
}

/**
 * This is the main entry point of a native application that is using
 * android_native_app_glue.  It runs in its own thread, with its own
 * event loop for receiving input events and doing other things.
 */
void android_main(struct android_app* state) {
    struct engine engine;

    // Make sure glue isn't stripped.
    app_dummy();

    memset(&engine, 0, sizeof(engine));
    state->userData = &engine;
    state->onAppCmd = engine_handle_cmd;
    engine.app = state;

    if (state->savedState != NULL) {
        // We are starting with a previous saved state; restore from it.
        engine.state = *(struct saved_state*)state->savedState;
    }

    // prepare the resource object
    r = new Resource(state);

    time_now = time_fps = now_s();

    while (1) {
        // Read all pending events.
        int ident;
        int events;
        struct android_poll_source* source;

        // If not animating, we will block forever waiting for events.
        // If animating, we loop until all events are read, then continue
        // to draw the next frame of animation.
        while ((ident=ALooper_pollAll(engine.animating ? 0 : -1, NULL, &events,
                (void**)&source)) >= 0) {

            // Process this event.
            if (source != NULL) {
                source->process(state, source);
            }

            // Check if we are exiting.
            if (state->destroyRequested != 0) {
                engine_term_display(&engine);
                return;
            }
        }

        if (engine.animating) {
            // Drawing is throttled to the screen update rate, so there
            // is no need to do timing here.
            engine_draw_frame(&engine);
        }
    }
}
//END_INCLUDE(all)
