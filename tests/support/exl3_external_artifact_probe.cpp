#include "exl3_external_artifacts.h"

int main(int argc, char** argv) {
  try {
    if (argc != 3) throw std::runtime_error("expected env/file/dir and a name");
    const std::string_view kind = argv[1];
    if (kind == "env") (void)exl3_test::ExternalEnvironment(argv[2]);
    else if (kind == "file" || kind == "dir")
      exl3_test::ExternalPath(argv[2], kind == "dir");
    else throw std::runtime_error("unknown artifact kind");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
