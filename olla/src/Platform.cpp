/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2008-2016 Stanford University and the Authors.      *
 * Authors: Peter Eastman                                                     *
 * Contributors:                                                              *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

#include "openmm/Platform.h"
#include "openmm/Context.h"
#include "openmm/OpenMMException.h"
#include "openmm/Kernel.h"
#include "openmm/KernelFactory.h"
#include "openmm/internal/ContextImpl.h"
#ifdef WIN32
#include <windows.h>
#else
#ifndef __PNACL__
    #include <dlfcn.h>
#endif
#include <dirent.h>
#include <cstdlib>
#endif
#include <sstream>
#include <set>
#include <algorithm>

#include "ReferencePlatform.h"

using namespace OpenMM;
using namespace std;

std::vector<std::string> Platform::pluginLoadFailures;
static bool stringLengthComparator(string i, string j) {
  return (i.size() < j.size());
}

static int registerPlatforms() {

    // Register the Platforms built into the main library.  This should eventually be moved elsewhere.

    ReferencePlatform* platform = new ReferencePlatform();
    Platform::registerPlatform(platform);
    return 0;
}

static int platformInitializer = registerPlatforms();

Platform::~Platform() {
    set<KernelFactory*> uniqueKernelFactories;
    for (auto& factory : kernelFactories)
        uniqueKernelFactories.insert(factory.second);
    for (auto factory : uniqueKernelFactories)
        delete factory;
}

const vector<string>& Platform::getPropertyNames() const {
    return platformProperties;
}

const string& Platform::getPropertyValue(const Context& context, const string& property) const {
    throw OpenMMException("getPropertyValue: Illegal property name");
}

void Platform::setPropertyValue(Context& context, const string& property, const string& value) const {
    throw OpenMMException("setPropertyValue: Illegal property name");
}

const string& Platform::getPropertyDefaultValue(const string& property) const {
    string propertyName = property;
    if (deprecatedPropertyReplacements.find(property) != deprecatedPropertyReplacements.end())
        propertyName = deprecatedPropertyReplacements.find(property)->second;
    map<string, string>::const_iterator value = defaultProperties.find(propertyName);
    if (value == defaultProperties.end())
        throw OpenMMException("getPropertyDefaultValue: Illegal property name");
    return value->second;
}

void Platform::setPropertyDefaultValue(const string& property, const string& value) {
    string propertyName = property;
    if (deprecatedPropertyReplacements.find(property) != deprecatedPropertyReplacements.end())
        propertyName = deprecatedPropertyReplacements.find(property)->second;
    for (auto& prop : platformProperties)
        if (prop == propertyName) {
            defaultProperties[propertyName] = value;
            return;
        }
    throw OpenMMException("setPropertyDefaultValue: Illegal property name");
}

void Platform::contextCreated(ContextImpl& context, const map<string, string>& properties) const {
}

void Platform::linkedContextCreated(ContextImpl& context, ContextImpl& originalContext) const {
    // The default implementation just copies over the properties and calls contextCreated().
    // Subclasses may override this to do something different.

    map<string, string> properties;
    for (auto& name : getPropertyNames())
        properties[name] = getPropertyValue(originalContext.getOwner(), name);
    contextCreated(context, properties);
}

void Platform::contextDestroyed(ContextImpl& context) const {
}

void Platform::registerKernelFactory(const string& name, KernelFactory* factory) {
    kernelFactories[name] = factory;
}

bool Platform::supportsKernels(const vector<string>& kernelNames) const {
    for (auto& name : kernelNames)
        if (kernelFactories.find(name) == kernelFactories.end())
            return false;
    return true;
}

Kernel Platform::createKernel(const string& name, ContextImpl& context) const {
    if (kernelFactories.find(name) == kernelFactories.end())
        throw OpenMMException("Called createKernel() on a Platform which does not support the requested kernel");
    return Kernel(kernelFactories.find(name)->second->createKernelImpl(name, *this, context));
}
vector<Platform*>& Platform::getPlatforms() {
    static vector<Platform*> platforms;
    return platforms;
}

void Platform::registerPlatform(Platform* platform) {
    getPlatforms().push_back(platform);
}

int Platform::getNumPlatforms() {
    return getPlatforms().size();
}

Platform& Platform::getPlatform(int index) {
    if (index >= 0 && index < getNumPlatforms()) {
        return *getPlatforms()[index];
    }
    throw OpenMMException("Invalid platform index");
}

std::vector<std::string> Platform::getPluginLoadFailures() {
  return pluginLoadFailures;
}

Platform& Platform::getPlatformByName(const string& name) {
    for (int i = 0; i < getNumPlatforms(); i++)
        if (getPlatform(i).getName() == name)
            return getPlatform(i);
    throw OpenMMException("There is no registered Platform called \""+name+"\"");
}

Platform& Platform::findPlatform(const vector<string>& kernelNames) {
    Platform* best = 0;
    vector<Platform*>& platforms = getPlatforms();
    double speed = 0.0;
    for (auto platform : platforms) {
        if (platform->supportsKernels(kernelNames) && platform->getSpeed() > speed) {
            best = platform;
            speed = best->getSpeed();
        }
    }
    if (best == 0)
        throw OpenMMException("No Platform supports all the requested kernels");
    return *best;
}

#ifdef WIN32
static HMODULE loadOneLibrary(const string& file) {
    // Tell Windows not to bother the user with ugly error boxes.
    const UINT oldErrorMode = SetErrorMode(SEM_FAILCRITICALERRORS);
    HMODULE handle = LoadLibrary(file.c_str());
    SetErrorMode(oldErrorMode); // Restore previous error mode.
    if (handle == NULL) {
        stringstream message;
        message << "Error loading library " << file << ": " << GetLastError();
        throw OpenMMException(message.str());
    }
    return handle;
}

static void initializePlugins(vector<HMODULE>& plugins) {
    for (auto plugin : plugins) {
        void (*init)();
        *(void **)(&init) = (void *) GetProcAddress(plugin, "registerPlatforms");
        if (init != NULL)
            (*init)();
    }
    for (auto plugin : plugins) {
        void (*init)();
        *(void **)(&init) = (void *) GetProcAddress(plugin, "registerKernelFactories");
        if (init != NULL)
            (*init)();
    }
}
#else
static void* loadOneLibrary(const string& file) {
#ifdef __PNACL__
    throw OpenMMException("Loading dynamic libraries is not supported on PNaCl");
#else
#ifdef __APPLE__
    void *handle = dlopen(file.c_str(), RTLD_LAZY | RTLD_GLOBAL);
#else
    void *handle = dlopen(file.c_str(), RTLD_LAZY | RTLD_LOCAL);
#endif
    if (handle == NULL) {
        throw OpenMMException("Error loading library "+file+": "+dlerror());
    }
    return handle;
#endif
}

static void initializePlugins(vector<void*>& plugins) {
#ifndef __PNACL__
    for (auto plugin : plugins) {
        void (*init)();
        *(void **)(&init) = dlsym(plugin, "registerPlatforms");
        if (init != NULL)
            (*init)();
    }
    for (auto plugin : plugins) {
        void (*init)();
        *(void **)(&init) = dlsym(plugin, "registerKernelFactories");
        if (init != NULL)
            (*init)();
    }
#endif
}
#endif

void Platform::loadPluginLibrary(const string& file) {
#ifdef WIN32
    vector<HMODULE> plugins;
#else
    vector<void*> plugins;
#endif
    plugins.push_back(loadOneLibrary(file));
    initializePlugins(plugins);
}

vector<string> Platform::loadPluginsFromDirectory(const string& directory) {
    vector<string> files;
    char dirSeparator;
    char pathSeparator;
    stringstream sdirectory(directory);
#ifdef WIN32
    dirSeparator = '\\';
    pathSeparator = ';';
    WIN32_FIND_DATA fileInfo;

    for (string path; std::getline(sdirectory, path, pathSeparator);) {
        string filePattern(path + dirSeparator + "*.dll");
        HANDLE findHandle = FindFirstFile(filePattern.c_str(), &fileInfo);
        if (findHandle != INVALID_HANDLE_VALUE) {
            do {
                if (fileInfo.cFileName[0] != '.')
                    files.push_back(path+dirSeparator+string(fileInfo.cFileName));
            } while (FindNextFile(findHandle, &fileInfo));
            FindClose(findHandle);
        }
    }
    vector<HMODULE> plugins;
#else
    DIR* dir;
    dirSeparator = '/';
    pathSeparator = ':';
    struct dirent *entry;

    for (string path; std::getline(sdirectory, path, pathSeparator);) {
        dir = opendir(path.c_str());
        if (dir != NULL) {
            while ((entry = readdir(dir)) != NULL) {
                if (entry->d_name[0] != '.')
                    files.push_back(path+dirSeparator+string(entry->d_name));
            }
            closedir(dir);
        }
    }

    vector<void*> plugins;
#endif
    vector<string> loadedLibraries;
    pluginLoadFailures.resize(0);
    std::sort (files.begin(), files.end(), stringLengthComparator);

    for (unsigned int i = 0; i < files.size(); ++i) {
        try {
            plugins.push_back(loadOneLibrary(files[i]));
            loadedLibraries.push_back(files[i]);
        } catch (OpenMMException& ex) {
	    pluginLoadFailures.push_back(ex.what());
        }
    }
    initializePlugins(plugins);
    return loadedLibraries;
}

const string& Platform::getDefaultPluginsDirectory() {
    char* dir = getenv("OPENMM_PLUGIN_DIR");
    static string directory;
#ifdef _MSC_VER
    if (dir != NULL)
        directory = string(dir);
    else {
        dir = getenv("PROGRAMFILES");
        if (dir == NULL)
            directory = "C:\\\\Program Files\\OpenMM\\lib\\plugins";
        else
            directory = string(dir)+"\\OpenMM\\lib\\plugins";
    }
#else
    if (dir == NULL)
        directory = OPENMM_INSTALL_PREFIX "/lib/plugins";
    else
        directory = string(dir);
#endif
    return directory;
}

// Some bizarre preprocessor magic required to convert a macro to a string...
#define STRING1(x) #x
#define STRING(x) STRING1(x)

const string& Platform::getOpenMMVersion() {
#if OPENMM_BUILD_VERSION == 0
    static const string version = STRING(OPENMM_MAJOR_VERSION) "." STRING(OPENMM_MINOR_VERSION);
#else
    static const string version = STRING(OPENMM_MAJOR_VERSION) "." STRING(OPENMM_MINOR_VERSION) "." STRING(OPENMM_BUILD_VERSION);
#endif
    return version;
}

ContextImpl& Platform::getContextImpl(Context& context) const {
    return *context.impl;
}

const ContextImpl& Platform::getContextImpl(const Context& context) const {
    return *context.impl;
}
