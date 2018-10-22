#include <iostream>
#include <string>

#include <console-common/console-plugin-base.h>
#include <map-manager/map-manager.h>
#include <vi-map/vi-map.h>

#include <octomap/octomap.h>
#include <Eigen/Geometry>
// Your new plugin needs to derive from ConsolePluginBase.
// (Alternatively, you can derive from ConsolePluginBaseWithPlotter if you need
// RViz plotting abilities for your VI map.)
class HelloWorldPlugin : public common::ConsolePluginBase
{
public:
  // Every plugin needs to implement a getPluginId function which returns a
  // string that gives each plugin a unique name.
  std::string getPluginId() const override
  {
    return "octomap_maplab_plugin";
  }

  // The constructor takes a pointer to the Console object which we can forward
  // to the constructor of ConsolePluginBase.
  HelloWorldPlugin(common::Console *console)
      : common::ConsolePluginBase(console)
  {
    // You can add your commands in here.
    addCommand(
        {"hello_world", "hello"}, // Map "hello_world" and "hello" to this
                                  // command.
        [this]() -> int {         // Function to call when this command is entered.
          // This function can do anything you want. Check the other plugins
          // under ~/maplab_ws/src/maplab/console-plugins for more examples.

          // Here, we just print a message to the terminal.
          std::cout << "Hello world!" << std::endl;

          // Every console command returns an integer, you can take one from
          // the CommandStatus enum. kSuccess returns everything is fine.
          // Other commonly used return values are common::kUnknownError and
          // common::kStupidUserError.
          return common::kSuccess;
        },

        // This is the description of your command. This will get printed when
        // you run `help` in the console.
        "This command prints \"Hello World!\" to the console.",

        // This specifies the execution method of your command. For most
        // commands, it is sufficient to run them in sync with
        // common::Processing::Sync.
        common::Processing::Sync);

    addCommand(
        {"convert_all_point_clouds_to_octomap"},

        [this]() -> int {
          // Get the currently selected map.
          std::string selected_map_key;

          // This function will write the name of the selected map key into
          // selected_map_key. The function will return false and print an error
          // message if no map key is selected.
          if (!getSelectedMapKeyIfSet(&selected_map_key))
          {
            return common::kStupidUserError;
          }

          // Create a map manager instance.
          vi_map::VIMapManager map_manager;

          // Get and lock the map which blocks all other access to the map.
          vi_map::VIMapManager::MapWriteAccess map =
              map_manager.getMapWriteAccess(selected_map_key);

          // Now run your algorithm on the VI map.
          // convert all optimized point clouds to octomap
          vi_map::VIMap *vi_map = map.get();

          // octomap tree
          octomap::OcTree tree(0.05);

          CHECK_NOTNULL(vi_map);
          vi_map->forEachVertex([&](vi_map::Vertex *vertex) {
            CHECK_NOTNULL(vertex);
            const vi_map::MissionId &mission_id = vertex->getMissionId();
            const aslam::NCamera &n_camera =
                vi_map->getSensorManager().getNCameraForMission(mission_id);

            const pose_graph::VertexId &vertex_id = vertex->id();
            const size_t num_frames = vertex->numFrames();
            const aslam::VisualNFrame &nframe = vertex->getVisualNFrame();
            for (size_t frame_idx = 0u; frame_idx < num_frames; ++frame_idx)
            {
              if (nframe.isFrameSet(frame_idx))
              {
                resources::PointCloud point_cloud;
                if (vi_map->getPointCloudXYZ(*vertex, frame_idx, &point_cloud))
                {
                  // Nothing to do here.
                }
                else
                {
                  continue;
                }
                CHECK(!point_cloud.empty()) << "Vertex " << vertex_id << " frame "
                                            << frame_idx << " has an empty point cloud!";
                // convert point_cloud in camera coordinate to world coordinate
                octomap::Pointcloud cloud;
                Eigen::Vector3d point;
                Eigen::Vector3d pointWorld;
                Eigen::Isometry3d T;
                for (size_t index = 0u; index < point_cloud.size(); index++)
                {
                  point[0] = point_cloud.xyz[0 + 3 * index];
                  point[1] = point_cloud.xyz[1 + 3 * index];
                  point[2] = point_cloud.xyz[2 + 3 * index];

                  pointWorld = T*point;
                  cloud.push_back(pointWorld[0], pointWorld[1], pointWorld[2]);
                }
                tree.insertPointCloud(cloud, octomap::point3d(T(0,3), T(1,3), T(2,3)));
              }
            }
          });
          // write octomap to disk
          tree.updateInnerOccupancy();
          tree.writeBinary( "octomap.bt" );

          return common::kSuccess;
        },

        "This command will run an awesome VI map algorithm.",
        common::Processing::Sync);
  }
};

// Finally, call the MAPLAB_CREATE_CONSOLE_PLUGIN macro to create your console
// plugin.
MAPLAB_CREATE_CONSOLE_PLUGIN(HelloWorldPlugin);
