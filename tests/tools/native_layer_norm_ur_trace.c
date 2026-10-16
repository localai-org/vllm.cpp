/* Diagnostic-only UR observer. Compile against the runtime's own ur_api.h.
 * It forwards every argument unchanged, captures bounded program images and
 * submitted geometry, and is never linked into product inference. */
#define _GNU_SOURCE
#include <stdbool.h>
#include <ur_api.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

static void *resolve(const char *name) {
  void *p = dlsym(RTLD_NEXT, name);
  if (!p) {
    /* Torch loads UR locally: use the already loaded runtime, never open a
     * second loader or forward handles into another implementation. */
    void *handle = dlopen("libur_loader.so.0", RTLD_LAZY | RTLD_NOLOAD);
    if (handle) { p = dlsym(handle, name); dlclose(handle); }
  }
  if (!p) fprintf(stderr, "N1 UR observer cannot resolve %s: %s\n", name, dlerror());
  return p;
}
static int active(void) {
  const char *p = getenv("N1_TRACE_ACTIVE");
  return p && strcmp(p, "1") == 0;
}
static FILE *logfile(void) {
  const char *dir = getenv("N1_TRACE_DIR");
  if (!dir) return NULL;
  char path[4096];
  int n = snprintf(path, sizeof(path), "%s/ur-trace.txt", dir);
  if (n < 0 || n >= (int)sizeof(path)) return NULL;
  return fopen(path, "a");
}
static void save(const void *bytes, size_t length, void *program, const char *kind) {
  static unsigned int count;
  if (!active() || length > 16 * 1024 * 1024 || !bytes) return;
  unsigned int id = __sync_fetch_and_add(&count, 1);
  if (id >= 32) return;
  const char *dir = getenv("N1_TRACE_DIR");
  if (!dir) return;
  char path[4096];
  int n = snprintf(path, sizeof(path), "%s/program-%u-%p.%s", dir, id, program, kind);
  if (n < 0 || n >= (int)sizeof(path)) return;
  FILE *f = fopen(path, "wb");
  if (f) { fwrite(bytes, 1, length, f); fclose(f); }
}
UR_APIEXPORT ur_result_t UR_APICALL urProgramCreateWithIL(ur_context_handle_t context,
    const void *bytes, size_t length, const ur_program_properties_t *props, ur_program_handle_t *out) {
  typedef ur_result_t (UR_APICALL *F)(ur_context_handle_t, const void *, size_t,
                                     const ur_program_properties_t *, ur_program_handle_t *);
  F next = (F)resolve("urProgramCreateWithIL");
  if (!next) return UR_RESULT_ERROR_UNINITIALIZED;
  ur_result_t result = next(context, bytes, length, props, out);
  if (result == UR_RESULT_SUCCESS) save(bytes, length, *out, "spv");
  return result;
}
UR_APIEXPORT ur_result_t UR_APICALL urProgramCreateWithBinary(ur_context_handle_t context,
    uint32_t count, ur_device_handle_t *devices, size_t *lengths, const uint8_t **bytes,
    const ur_program_properties_t *props, ur_program_handle_t *out) {
  typedef ur_result_t (UR_APICALL *F)(ur_context_handle_t, uint32_t, ur_device_handle_t *, size_t *,
                                     const uint8_t **, const ur_program_properties_t *, ur_program_handle_t *);
  F next = (F)resolve("urProgramCreateWithBinary");
  if (!next) return UR_RESULT_ERROR_UNINITIALIZED;
  ur_result_t result = next(context, count, devices, lengths, bytes, props, out);
  if (result == UR_RESULT_SUCCESS)
    for (uint32_t i = 0; i < count && i < 4; ++i) save(bytes[i], lengths[i], *out, "bin");
  return result;
}
UR_APIEXPORT ur_result_t UR_APICALL urKernelCreate(ur_program_handle_t program, const char *name,
                                                ur_kernel_handle_t *out) {
  typedef ur_result_t (UR_APICALL *F)(ur_program_handle_t, const char *, ur_kernel_handle_t *);
  F next = (F)resolve("urKernelCreate");
  if (!next) return UR_RESULT_ERROR_UNINITIALIZED;
  ur_result_t result = next(program, name, out);
  if (active()) {
    FILE *f = logfile();
    if (f) { fprintf(f, "KERNEL program=%p kernel=%p result=%d name=%s\n",
                     (void *)program, result == UR_RESULT_SUCCESS ? (void *)*out : NULL, result, name); fclose(f); }
  }
  return result;
}
UR_APIEXPORT ur_result_t UR_APICALL urEnqueueKernelLaunch(ur_queue_handle_t queue, ur_kernel_handle_t kernel,
    uint32_t dim, const size_t *offset, const size_t *global, const size_t *local,
    const ur_kernel_launch_ext_properties_t *props, uint32_t count,
    const ur_event_handle_t *wait, ur_event_handle_t *event) {
  typedef ur_result_t (UR_APICALL *F)(ur_queue_handle_t, ur_kernel_handle_t, uint32_t,
      const size_t *, const size_t *, const size_t *, const ur_kernel_launch_ext_properties_t *,
      uint32_t, const ur_event_handle_t *, ur_event_handle_t *);
  F next = (F)resolve("urEnqueueKernelLaunch");
  if (!next) return UR_RESULT_ERROR_UNINITIALIZED;
  if (active()) {
    FILE *f = logfile();
    if (f) {
      fprintf(f, "LAUNCH kernel=%p dims=%u global=", (void *)kernel, dim);
      for (uint32_t i = 0; i < dim && i < 3; ++i) fprintf(f, "%zu,", global[i]);
      fprintf(f, " local=");
      for (uint32_t i = 0; i < dim && i < 3; ++i) fprintf(f, "%zu,", local ? local[i] : 0);
      fprintf(f, "\n"); fclose(f);
    }
  }
  return next(queue, kernel, dim, offset, global, local, props, count, wait, event);
}
UR_APIEXPORT ur_result_t UR_APICALL urEnqueueKernelLaunchWithArgsExp(
    ur_queue_handle_t queue, ur_kernel_handle_t kernel, uint32_t dim,
    const size_t *offset, const size_t *global, const size_t *local, uint32_t numArgs,
    const ur_exp_kernel_arg_properties_t *args, const ur_kernel_launch_ext_properties_t *props,
    uint32_t count, const ur_event_handle_t *wait, ur_event_handle_t *event) {
  typedef ur_result_t (UR_APICALL *F)(ur_queue_handle_t, ur_kernel_handle_t, uint32_t,
      const size_t *, const size_t *, const size_t *, uint32_t, const ur_exp_kernel_arg_properties_t *,
      const ur_kernel_launch_ext_properties_t *, uint32_t, const ur_event_handle_t *, ur_event_handle_t *);
  F next = (F)resolve("urEnqueueKernelLaunchWithArgsExp");
  if (!next) return UR_RESULT_ERROR_UNINITIALIZED;
  if (active()) {
    FILE *f = logfile();
    if (f) {
      fprintf(f, "LAUNCH_ARGS kernel=%p dims=%u global=", (void *)kernel, dim);
      for (uint32_t i = 0; i < dim && i < 3; ++i) fprintf(f, "%zu,", global[i]);
      fprintf(f, " local=");
      for (uint32_t i = 0; i < dim && i < 3; ++i) fprintf(f, "%zu,", local ? local[i] : 0);
      fprintf(f, "\n"); fclose(f);
    }
  }
  return next(queue, kernel, dim, offset, global, local, numArgs, args, props, count, wait, event);
}
UR_APIEXPORT ur_result_t UR_APICALL urKernelGetGroupInfo(ur_kernel_handle_t kernel,
    ur_device_handle_t device, ur_kernel_group_info_t param, size_t size, void *value, size_t *actual) {
  typedef ur_result_t (UR_APICALL *F)(ur_kernel_handle_t, ur_device_handle_t, ur_kernel_group_info_t,
                                    size_t, void *, size_t *);
  F next = (F)resolve("urKernelGetGroupInfo");
  if (!next) return UR_RESULT_ERROR_UNINITIALIZED;
  ur_result_t result = next(kernel, device, param, size, value, actual);
  if (active() && value && result == UR_RESULT_SUCCESS) {
    typedef ur_result_t (UR_APICALL *D)(ur_device_handle_t, ur_device_info_t, size_t, void *, size_t *);
    D info = (D)resolve("urDeviceGetInfo");
    uint32_t ip = 0;
    if (info && info(device, UR_DEVICE_INFO_IP_VERSION, sizeof(ip), &ip, NULL) == UR_RESULT_SUCCESS) {
      FILE *f = logfile();
      if (f) { fprintf(f, "DEVICE_IP value=%u hex=%08x\n", ip, ip); fclose(f); }
    }

    FILE *f = logfile();
    if (f) {
      fprintf(f, "GROUP_INFO kernel=%p param=%u bytes=%zu value=", (void *)kernel, param, size);
      for (size_t i = 0; i < size && i < 24; ++i) fprintf(f, "%02x", ((unsigned char *)value)[i]);
      fprintf(f, "\n"); fclose(f);
    }
  }
  return result;
}
UR_APIEXPORT ur_result_t UR_APICALL urProgramBuild(ur_context_handle_t context,
    ur_program_handle_t program, const char *options) {
  typedef ur_result_t (UR_APICALL *F)(ur_context_handle_t, ur_program_handle_t, const char *);
  F next = (F)resolve("urProgramBuild");
  if (!next) return UR_RESULT_ERROR_UNINITIALIZED;
  if (active()) {
    FILE *f = logfile();
    if (f) { fprintf(f, "BUILD program=%p options=%s\n", (void *)program, options ? options : "<null>"); fclose(f); }
  }
  return next(context, program, options);
}
UR_APIEXPORT ur_result_t UR_APICALL urProgramBuildExp(ur_program_handle_t program,
    uint32_t count, ur_device_handle_t *devices, ur_exp_program_flags_t flags, const char *options) {
  typedef ur_result_t (UR_APICALL *F)(ur_program_handle_t, uint32_t, ur_device_handle_t *,
                                    ur_exp_program_flags_t, const char *);
  F next = (F)resolve("urProgramBuildExp");
  if (!next) return UR_RESULT_ERROR_UNINITIALIZED;
  if (active()) {
    FILE *f = logfile();
    if (f) { fprintf(f, "BUILD_EXP program=%p options=%s\n", (void *)program, options ? options : "<null>"); fclose(f); }
  }
  return next(program, count, devices, flags, options);
}
