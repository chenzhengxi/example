/** @file */

#pragma once
#include <string>

/// Namespace of the Plugin library
namespace Plugin
{
  /// Interface of a plugin
  /**
      * You are supposed to inherit and implement this interface in your plugin facade.
      * Your facade factory should be created using the macro
      * PLUGIN_FACTORY_DECLARATION(T) and PLUGIN_FACTORY_DEFINITION(T).
      */
  class IPlugin
  {
  public:
    IPlugin() = default;
    IPlugin(const IPlugin &) = delete;
    void operator=(const IPlugin &c) = delete;
    /// Get plugin name
    virtual const std::string &iGetPluginName() const = 0;
    virtual const std::string &iWhat(int value, const std::string &msg) const = 0;
    /// Get plugin version
    //virtual const Vers::Version& iGetPluginVersion() const = 0;

  protected:
    /// Destructor
    /**
          * This destructor is declared protected
          * in order to strictly forbid user
          * to delete instance of a plugin facade.
          * Construction and destruction of a plugin facade
          * must all resides inside the boundary
          * of the concrete plugin library.
          */
    virtual ~IPlugin() {}
  };
} // namespace Plugin
