// -*- C++ -*-
#ifndef OBJIMPORTER_HEADER
#define OBJIMPORTER_HEADER

class MeshData;

class ObjImporter {
public:
    static MeshData* import(const char* path);
};

#endif
