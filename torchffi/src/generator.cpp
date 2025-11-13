#include <torch/all.h>
#include <torch_ffi.h>

#include <optional>

Generator torchffi_get_default_generator(Device* device) {
  at::Device cDevice = device ? at::Device(at::DeviceType(device->type), device->index) : at::DeviceType::CPU;
  auto generator = at::globalContext().defaultGenerator(cDevice);
  return new torch::Generator(generator);
}

Generator torchffi_generator_new(Device* device) {
  auto generator = at::globalContext().getAcceleratorHooksInterface(
                                        device ? std::optional<c10::DeviceType>(at::DeviceType(device->type)) : std::nullopt
  )
                     .getNewGenerator(device ? device->index : -1);
  return new torch::Generator(generator);
}

/*
Generator torchffi_generator_new_cpu(uint64_t seed) {
  return new torch::Generator(at::detail::createCPUGenerator(seed));
}*/

Generator torchffi_generator_clone(Generator generator) {
  return new torch::Generator(generator->clone());
}

void torchffi_generator_delete(Generator generator) {
  delete generator;
}

void torchffi_generator_set_current_seed(Generator generator, uint64_t seed) {
  generator->set_current_seed(seed);
}

uint64_t torchffi_generator_get_current_seed(Generator generator) {
  return generator->current_seed();
}

void torchffi_generator_set_offset(Generator generator, uint64_t offset) {
  generator->set_offset(offset);
}

uint64_t torchffi_generator_get_offset(Generator generator) {
  return generator->get_offset();
}

void torchffi_generator_set_state(Generator generator, tensor newState) {
  generator->set_state(*newState);
}

tensor torchffi_generator_get_state(Generator generator) {
  return new torch::Tensor(generator->get_state());
}

Device torchffi_generator_get_device(Generator generator) {
  auto device = generator->device();
  return Device{int8_t(device.type()), device.index()};
}