#ifndef __AssetLoader__
#define __AssetLoader__

#include <android_native_app_glue.h>
#include <GLES3/gl3.h>
#include <vector>

class Resource
{
public:
  Resource(android_app* pApplication);
  GLuint read(const char* m_path, std::vector<unsigned char>& buffer);

private:
  AAssetManager* m_assetManager;
};
#endif