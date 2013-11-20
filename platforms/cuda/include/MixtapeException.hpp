#include <exception>
#include <string>
namespace Mixtape {

class MixtapeException : public std::exception {
public:
    explicit MixtapeException(const std::string& message) : message(message) {
    }
    ~MixtapeException() throw() {
    }
    const char* what() const throw() {
        return message.c_str();
    }
private:
    std::string message;
};

}  // namespace Mixtape
