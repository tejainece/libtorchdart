#include <torch_ffi.h>

int main(int argc, char *argv[]) {
    tensor t = torchffi_new_tensor();
    torchffi_new_tensor_eye(t, );
    return 0;
}