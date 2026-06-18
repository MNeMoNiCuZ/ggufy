FROM debian:bookworm-slim AS zig-download

ARG ZIG_VERSION=0.16.0
ARG ZIG_MINISIG_PK=RWSGOq2NVecA2UPNdBUZykf1CCb147pkmdtYxgb3Ti+JO/wCYvhbAb/U

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        minisign \
        xz-utils \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/zig && \
    curl -L "https://ziglang.org/download/${ZIG_VERSION}/zig-x86_64-linux-${ZIG_VERSION}.tar.xz" -o /tmp/zig.tar.xz && \
    curl -L "https://ziglang.org/download/${ZIG_VERSION}/zig-x86_64-linux-${ZIG_VERSION}.tar.xz.minisig" -o /tmp/zig.tar.xz.minisig && \
    minisign -Vm /tmp/zig.tar.xz -P "$ZIG_MINISIG_PK" && \
    tar -xJf /tmp/zig.tar.xz -C /opt/zig --strip-components=1 && \
    rm /tmp/zig.tar.xz /tmp/zig.tar.xz.minisig

FROM debian:bookworm-slim AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=zig-download /opt/zig /opt/zig

ENV PATH=/opt/zig:${PATH}

WORKDIR /opt/ggufy
COPY . /opt/ggufy
RUN zig build --release=fast

FROM debian:bookworm-slim AS runtime

WORKDIR /app
COPY --from=builder /opt/ggufy/zig-out/bin/ggufy /usr/local/bin/ggufy

ENTRYPOINT ["ggufy"]
CMD ["--help"]
