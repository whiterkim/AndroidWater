//
//  Texture.cpp
//  libpngMem
//
//  Created by Fakhir Shaheen on 02/12/2014.
//  Copyright (c) 2014 Fakhir Shaheen. All rights reserved.
//

#include "AssetLoader.h"

Resource::Resource(android_app* pApplication)
          :m_assetManager
          (pApplication->activity->assetManager){
}

GLuint Resource::read(const char* m_path, std::vector<unsigned char> & buffer){
  AAsset* m_asset = AAssetManager_open(m_assetManager, m_path, AASSET_MODE_UNKNOWN);
  if(m_asset == NULL) return GL_FALSE;

  int size = AAsset_getLength(m_asset);
  buffer.resize(size);
  int32_t count = AAsset_read (m_asset, &buffer[0], size);
  if(count != size) return GL_FALSE;

  AAsset_close(m_asset);
  return size;
}