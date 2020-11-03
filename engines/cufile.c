/*
 * Copyright (c)2020 System Fabric Works, Inc. All Rights Reserved.
 * mailto:info@systemfabricworks.com
 *
 * License: GPLv2, see COPYING.
 *
 * cufile engine
 *
 * fio I/O engine using the NVIDIA cuFile API.
 *
 */

#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <cufile.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pthread.h>

#include "../fio.h"
#include "../lib/pow2.h"
#include "../optgroup.h"
#include "../lib/memalign.h"

#define ALIGNED_4KB(v) (((v) & 0x0fff) == 0)

#define LOGGED_BUFLEN_NOT_ALIGNED     0x01
#define LOGGED_OFFSET_NOT_ALIGNED     0x02
#define LOGGED_GPU_OFFSET_NOT_ALIGNED 0x03
#define GPU_ID_SEP ":"

struct cufile_options {
	struct thread_data *td;
	char               *gpu_ids;       /* colon-separated list of GPU ids,
										  one per job */
	void               *cu_mem_ptr;    /* GPU memory */
	void               *junk_buf;      /* buffer to simulate cudaMemcpy with
										  posix I/O write */
	int                 my_gpu_id;     /* GPU id to use for this job */
	unsigned int        use_posixio;   /* Use POSIX I/O instead of cuFile API */
	size_t              total_mem;     /* size for cu_mem_ptr and junk_buf */
	int                 logged;        /* bitmask of log messages that have
										  been output, prevent flood */
};

struct fio_cufile_data {
	CUfileDescr_t  cf_descr;
	CUfileHandle_t cf_handle;
};

struct fio_option options[] = {
	{
		.name	= "cufile_gpu_ids",
		.lname	= "cufile gpu ids",
		.type	= FIO_OPT_STR_STORE,
		.off1	= offsetof(struct cufile_options, gpu_ids),
		.help	= "GPU IDs, one per job, separated by " GPU_ID_SEP,
		.category = FIO_OPT_C_ENGINE,
		.group	= FIO_OPT_G_CUFILE,
	},
	{
		.name	= "cufile_posixio",
		.lname	= "cufile posixio",
		.type	= FIO_OPT_BOOL,
		.off1	= offsetof(struct cufile_options, use_posixio),
		.help	= "Use POSIX I/O instead of cuFile API",
		.def    = "0",
		.category = FIO_OPT_C_ENGINE,
		.group	= FIO_OPT_G_CUFILE,
	},
	{
		.name	= NULL,
	},
};

static int running = 0;
static int cufile_initialized = 0;
static pthread_mutex_t running_lock = PTHREAD_MUTEX_INITIALIZER;

#define check_cudaruntimecall(fn, rc)								\
	do {															\
		cudaError_t res = fn;										\
		if (res != cudaSuccess) {									\
			const char *str = cudaGetErrorName(res);				\
			log_err("cuda runtime api call failed %s:%d : %s\n",	\
					#fn, __LINE__, str);							\
			rc = -1;												\
		} else {													\
			rc = 0;													\
		}															\
	} while(0)

static const char* cuFileGetErrorString(CUfileError_t st)
{
	if (IS_CUFILE_ERR(st.err)) {
		return cufileop_status_error(st.err);
	}
	return "unknown";
}

/**
 * fio_cufile_init called in job process after fork() or pthread_create().
 */
static int fio_cufile_init(struct thread_data *td)
{
	struct cufile_options *o = td->eo;
	char* cur = NULL;
	char* pos = NULL;
	CUfileError_t status;
	int gpu_id = 0;
	int i;
	int initialized;
	int rc;

	pthread_mutex_lock(&running_lock);
	if (running == 0) {
		assert(cufile_initialized == 0);
		if (!o->use_posixio) {
			/* only open the driver if this is the first worker thread */
			status = cuFileDriverOpen();
			if (status.err != CU_FILE_SUCCESS) {
				log_err("cuFileDriverOpen: %d:%s\n", status.err,
						cuFileGetErrorString(status));
			} else {
				cufile_initialized = 1;
			}
		}
	}
	running++;
	initialized = cufile_initialized;
	pthread_mutex_unlock(&running_lock);

	if (!o->use_posixio && !initialized) {
		return 1;
	}

	if (o->gpu_ids) {
		for (i = 0, cur = strtok_r(o->gpu_ids, GPU_ID_SEP, &pos);
			 i <= td->subjob_number && cur;
			 ++i, cur = strtok_r(NULL, GPU_ID_SEP, &pos)) {
			gpu_id = atoi(cur);
		}
	}

	o->my_gpu_id = gpu_id;

	check_cudaruntimecall(cudaSetDevice(o->my_gpu_id), rc);
	if (rc != 0) {
		return 1;
	}

	return 0;
}

static inline int fio_cufile_pre_write(struct thread_data *td,
									   struct cufile_options *o,
									   struct io_u *io_u,
									   size_t gpu_offset)
{
	int rc = 0;

	if (!o->use_posixio) {
		if (td->o.verify) {
			/*
			  Data is being verified, copy the io_u buffer to GPU memory.
			  This isn't done in the non-verify case because the data would
			  already be in GPU memory in a normal cuFile application.
			*/
			check_cudaruntimecall(cudaMemcpy(((char*) o->cu_mem_ptr) + gpu_offset,
											 io_u->xfer_buf,
											 io_u->xfer_buflen,
											 cudaMemcpyHostToDevice), rc);
			if (rc != 0) {
				log_err("DDIR_WRITE cudaMemcpy H2D: %d\n", rc);
				io_u->error = EIO;
			}
		}
	} else {

		/*
		  POSIX I/O is being used, the data has to be copied out of the
		  GPU into a CPU buffer. GPU memory doesn't contain the actual
		  data to write, copy the data to the junk buffer. The purpose
		  of this is to add the overhead of cudaMemcpy() that would be
		  present in a POSIX I/O CUDA application.
		*/
		check_cudaruntimecall(cudaMemcpy(o->junk_buf + gpu_offset,
										 ((char*) o->cu_mem_ptr) + gpu_offset,
										 io_u->xfer_buflen,
										 cudaMemcpyDeviceToHost), rc);
		if (rc != 0) {
			if (rc != 0) {
				log_err("DDIR_WRITE cudaMemcpy D2H: %d\n", rc);
				io_u->error = EIO;
			}
		}
	}

	return rc;
}

static inline int fio_cufile_post_read(struct thread_data *td,
									   struct cufile_options *o,
									   struct io_u *io_u,
									   size_t gpu_offset)
{
	int rc = 0;

	if (!o->use_posixio) {
		if (td->o.verify) {
			/* Copy GPU memory to CPU buffer for verify */
			check_cudaruntimecall(cudaMemcpy(io_u->xfer_buf,
											 ((char*) o->cu_mem_ptr) + gpu_offset,
											 io_u->xfer_buflen,
											 cudaMemcpyDeviceToHost), rc);
			if (rc != 0) {
				log_err("DDIR_READ cudaMemcpy D2H: %d\n", rc);
				io_u->error = EIO;
			}
		}
	} else {
		/* POSIX I/O read, copy the CPU buffer to GPU memory */
		check_cudaruntimecall(cudaMemcpy(((char*) o->cu_mem_ptr) + gpu_offset,
										 io_u->xfer_buf,
										 io_u->xfer_buflen,
										 cudaMemcpyHostToDevice), rc);
		if (rc != 0) {
			log_err("DDIR_READ cudaMemcpy H2D: %d\n", rc);
			io_u->error = EIO;
		}
	}

	return rc;
}

static enum fio_q_status fio_cufile_queue(struct thread_data *td,
										  struct io_u *io_u)
{
	struct cufile_options *o = td->eo;
	struct fio_cufile_data *fcd = FILE_ENG_DATA(io_u->file);
	unsigned long long io_offset;
	ssize_t sz;
	ssize_t remaining;
	size_t xfered;
	size_t gpu_offset;
	int rc;

	if (!o->use_posixio && fcd == NULL) {
		io_u->error = EINVAL;
		td_verror(td, EINVAL, "xfer");
		return FIO_Q_COMPLETED;
	}

	fio_ro_check(td, io_u);

	switch(io_u->ddir) {
	case DDIR_SYNC:
		rc = fsync(io_u->file->fd);
		if (rc != 0) {
			io_u->error = errno;
			log_err("fsync: %d\n", errno);
		}
		break;

	case DDIR_DATASYNC:
		rc = fdatasync(io_u->file->fd);
		if (rc != 0) {
			io_u->error = errno;
			log_err("fdatasync: %d\n", errno);
		}
		break;

	case DDIR_READ:
	case DDIR_WRITE:
		/*
		  There may be a better way to calculate gpu_offset. The intent is
		  that gpu_offset equals the the difference between io_u->xfer_buf and
		  the page-aligned base address for io_u buffers.
		*/
		gpu_offset = io_u->index * io_u->xfer_buflen;
		io_offset = io_u->offset;
		remaining = io_u->xfer_buflen;

		xfered = 0;
		sz = 0;

		assert(gpu_offset + io_u->xfer_buflen <= o->total_mem);

		if (!(ALIGNED_4KB(io_u->xfer_buflen) ||
			  (o->logged & LOGGED_BUFLEN_NOT_ALIGNED))) {
			log_err("buflen not 4K-aligned: %llu\n", io_u->xfer_buflen);
			o->logged |= LOGGED_BUFLEN_NOT_ALIGNED;
		}

		if (!(ALIGNED_4KB(gpu_offset) ||
			  (o->logged & LOGGED_GPU_OFFSET_NOT_ALIGNED))) {
			log_err("gpu_offset not 4K-aligned: %lu\n", gpu_offset);
			o->logged |= LOGGED_GPU_OFFSET_NOT_ALIGNED;
		}

		if (io_u->ddir == DDIR_WRITE) {
			rc = fio_cufile_pre_write(td, o, io_u, gpu_offset);
		}

		if (io_u->error != 0) {
			break;
		}

		while (remaining > 0) {
			assert(gpu_offset + xfered <= o->total_mem);
			if (io_u->ddir == DDIR_READ) {
				if (!o->use_posixio) {
					sz = cuFileRead(fcd->cf_handle, o->cu_mem_ptr, remaining,
									io_offset + xfered, gpu_offset + xfered);
					if (sz == -1) {
						io_u->error = errno;
						log_err("cuFileRead: errno=%d\n", errno);
					} else if (sz < 0) {
						io_u->error = EIO;
						log_err("cuFileRead: %ld:%s\n", sz,
								cufileop_status_error(-sz));
					}
				} else {
					sz = pread(io_u->file->fd, ((char*) io_u->xfer_buf) + xfered,
							   remaining, io_offset + xfered);
					if (sz < 0) {
						io_u->error = errno;
						log_err("pread: %d\n", errno);
					}
				}
			} else if (io_u->ddir == DDIR_WRITE) {
				if (!o->use_posixio) {
					sz = cuFileWrite(fcd->cf_handle, o->cu_mem_ptr, remaining,
									 io_offset + xfered, gpu_offset + xfered);
					if (sz == -1) {
						io_u->error = errno;
						log_err("cuFileWrite: errno=%d\n", errno);
					} else if (sz < 0) {
						io_u->error = EIO;
						log_err("cuFileWrite: %ld:%s\n", sz,
								cufileop_status_error(-sz));
					}
				} else {
					sz = pwrite(io_u->file->fd,
								((char*) io_u->xfer_buf) + xfered,
								remaining, io_offset + xfered);
					if (sz < 0) {
						io_u->error = errno;
						log_err("pwrite: %d\n", errno);
					}
				}
			} else {
				/* shouldn't happen */
				assert(0);
				log_err("not DDIR_READ or DDIR_WRITE: %d\n", io_u->ddir);
				io_u->error = -1;
				break;
			}

			if (io_u->error != 0) {
				break;
			}

			remaining -= sz;
			xfered += sz;

			if (remaining != 0) {
				log_info("Incomplete %s: %ld bytes remaining\n",
						 io_u->ddir == DDIR_READ? "read" : "write", remaining);
			}

		}

		if (io_u->error != 0) {
			break;
		}

		if (io_u->ddir == DDIR_READ) {
			rc = fio_cufile_post_read(td, o, io_u, gpu_offset);
		}
		break;

	default:
		io_u->error = EINVAL;
		break;
	}

	if (io_u->error != 0) {
		log_err("IO failed\n");
		td_verror(td, io_u->error, "xfer");
	}

	return FIO_Q_COMPLETED;
}

static int fio_cufile_open_file(struct thread_data *td, struct fio_file *f)
{
	struct cufile_options *o = td->eo;
	struct fio_cufile_data *fcd = NULL;
	int rc;
	CUfileError_t status;

	rc = generic_open_file(td, f);
	if (rc) {
		return rc;
	}

	if (!o->use_posixio) {
		fcd = calloc(1, sizeof(*fcd));
		if (!fcd) {
			rc = ENOMEM;
			goto exit_err;
		}

		fcd->cf_descr.handle.fd = f->fd;
		fcd->cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
		status = cuFileHandleRegister(&fcd->cf_handle, &fcd->cf_descr);
		if (status.err != CU_FILE_SUCCESS) {
			log_err("cufile register error: %d:%s\n", status.err,
					cuFileGetErrorString(status));
			rc = EINVAL;
			goto exit_err;
		}
	}

	FILE_SET_ENG_DATA(f, fcd);
	return 0;

exit_err:
	if (fcd) {
		free(fcd);
		fcd = NULL;
	}
	if (f) {
		int rc2 = generic_close_file(td, f);
		if (rc2)
			log_err("generic_close_file %s:%d : %d\n", __func__, __LINE__, rc2);
	}
	return rc;
}

static int fio_cufile_close_file(struct thread_data *td, struct fio_file *f)
{
	struct fio_cufile_data *fcd = FILE_ENG_DATA(f);
	int rc;

	if (fcd != NULL) {
		cuFileHandleDeregister(fcd->cf_handle);
		FILE_SET_ENG_DATA(f, NULL);
		free(fcd);
	}

	rc = generic_close_file(td, f);

	return rc;
}

static int fio_cufile_iomem_alloc(struct thread_data *td, size_t total_mem)
{
	struct cufile_options *o = td->eo;
	int rc;
	CUfileError_t status;

	o->total_mem = total_mem;
	o->logged = 0;
	td->orig_buffer = calloc(1, total_mem);
	if (!td->orig_buffer) {
		log_err("calloc failed: %d\n", errno);
		goto exit_error;
	}

	if (o->use_posixio) {
		o->junk_buf = calloc(1, total_mem);
		if (o->junk_buf == NULL) {
			goto exit_error;
		}
	} else {
		o->junk_buf = NULL;
	}

	dprint(FD_MEM, "Alloc %zu for GPU %d\n", total_mem, o->my_gpu_id);
	check_cudaruntimecall(cudaMalloc(&o->cu_mem_ptr, total_mem), rc);
	if (rc != 0) {
		goto exit_error;
	}
	check_cudaruntimecall(cudaMemset(o->cu_mem_ptr, 0xab, total_mem), rc);
	if (rc != 0) {
		goto exit_error;
	}

	if (!o->use_posixio) {
		status = cuFileBufRegister(o->cu_mem_ptr, total_mem, 0);
		if (status.err != CU_FILE_SUCCESS) {
			log_err("cuFileBufRegister: %d:%s\n", status.err,
					cuFileGetErrorString(status));
			goto exit_error;
		}
	}

	return 0;

exit_error:
	if (td->orig_buffer) {
		free(td->orig_buffer);
		td->orig_buffer = NULL;
	}
	if (o->junk_buf) {
		free(o->junk_buf);
		o->junk_buf = NULL;
	}
	if (o->cu_mem_ptr) {
		cudaFree(o->cu_mem_ptr);
		o->cu_mem_ptr = NULL;
	}
	return 1;
}

static void fio_cufile_iomem_free(struct thread_data *td)
{
	struct cufile_options *o = td->eo;
	if (o->junk_buf) {
		free(o->junk_buf);
		o->junk_buf = NULL;
	}
	if (o->cu_mem_ptr) {
		if (!o->use_posixio) {
			cuFileBufDeregister(o->cu_mem_ptr);
		}
		cudaFree(o->cu_mem_ptr);
		o->cu_mem_ptr = NULL;
	}
	if (td->orig_buffer) {
		free(td->orig_buffer);
		td->orig_buffer = NULL;
	}
}

static void fio_cufile_cleanup(struct thread_data *td)
{
	struct cufile_options *o = td->eo;
	pthread_mutex_lock(&running_lock);
	running--;
	assert(running >= 0);
	if (running == 0) {
		/* only close the driver if initialized and
		   this is the last worker thread */
		if (!o->use_posixio && cufile_initialized) {
			cuFileDriverClose();
		}
		cufile_initialized = 0;
	}
	pthread_mutex_unlock(&running_lock);
}

FIO_STATIC struct ioengine_ops ioengine = {
	.name                = "cufile",
	.version             = FIO_IOOPS_VERSION,
	.init                = fio_cufile_init,
	.queue               = fio_cufile_queue,
	.open_file           = fio_cufile_open_file,
	.close_file          = fio_cufile_close_file,
	.iomem_alloc         = fio_cufile_iomem_alloc,
	.iomem_free          = fio_cufile_iomem_free,
	.cleanup             = fio_cufile_cleanup,
	.flags               = FIO_SYNCIO,
	.options             = options,
	.option_struct_size  = sizeof(struct cufile_options)
};

void fio_init fio_cufile_register(void)
{
	const char* diskless = getenv("FIO_CUFILE_DISKLESSIO");
	if ((diskless != NULL) && !strcmp(diskless, "1")) {
		ioengine.flags |= FIO_DISKLESSIO;
	}
	register_ioengine(&ioengine);
}

void fio_exit fio_cufile_unregister(void)
{
	unregister_ioengine(&ioengine);
}
