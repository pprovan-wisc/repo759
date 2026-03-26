#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <mpi.h>

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size != 2) {
    if (rank == 0) fprintf(stderr, "Requires exactly 2 MPI processes.\n");
    MPI_Finalize();
    return 1;
  }

  if (argc < 3) {
    if (rank == 0) fprintf(stderr, "Usage: %s n outfile\n", argv[0]);
    MPI_Finalize();
    return 1;
  }

  const int n        = std::atoi(argv[1]);
  const char *outfile = argv[2];

  float *send_buf = new float[n];
  float *recv_buf = new float[n];

  for (int i = 0; i < n; i++)
    send_buf[i] = static_cast<float>(rank * n + i);

  const int tag = 0;

  MPI_Barrier(MPI_COMM_WORLD);

  double t0_ms = 0.0, t1_ms = 0.0;

  if (rank == 0) {
    auto start = std::chrono::high_resolution_clock::now();
    MPI_Send(send_buf, n, MPI_FLOAT, 1, tag, MPI_COMM_WORLD);
    MPI_Recv(recv_buf, n, MPI_FLOAT, 1, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    auto end = std::chrono::high_resolution_clock::now();
    t0_ms = std::chrono::duration<double, std::milli>(end - start).count();

    MPI_Recv(&t1_ms, 1, MPI_DOUBLE, 1, tag + 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    FILE *f = fopen(outfile, "w");
    if (f) {
      fprintf(f, "%.6f\n", t0_ms + t1_ms);
      fclose(f);
    }

  } else {
    auto start = std::chrono::high_resolution_clock::now();
    MPI_Recv(recv_buf, n, MPI_FLOAT, 0, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Send(send_buf, n, MPI_FLOAT, 0, tag, MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    t1_ms = std::chrono::duration<double, std::milli>(end - start).count();

    MPI_Send(&t1_ms, 1, MPI_DOUBLE, 0, tag + 1, MPI_COMM_WORLD);
  }

  delete[] send_buf;
  delete[] recv_buf;

  MPI_Finalize();
  return 0;
}
