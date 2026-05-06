# CLOUD COMPUTING — COMPREHENSIVE PYQ ANALYSIS REPORT
### PES University | UE18/19/20/21/22CS351/352 | Papers: 2021, 2022, 2023 (July & Dec), 2024, 2025

---

> **Papers Analysed:** July 2021 · May 2022 · July 2023 · Dec 2023 · Jan–May 2024 · Dec 2025  
> **Total Questions Extracted:** 68  
> **Syllabus Units:** 4  
> **Max Marks per Paper:** 100

---

## STEP 1 — COMPLETE QUESTION EXTRACTION

### ─── 2021 (July 2021) ───

| Q.No | Exact Question | Marks | Unit | Main Topic | Subtopic | Type | Difficulty |
|------|----------------|-------|------|------------|----------|------|------------|
| 1a | The International Data Corporation has found that 80% of organisations have moved applications out of public cloud and back to private cloud. Why are companies moving certain applications and workloads back to private cloud? What are the issues you might encounter if you go "all in" with public cloud? | 5 | 1 | Deployment Models | Public vs Private Cloud | Explain | Medium |
| 1b | Imagine if you are developing the back-end service for a website like codechef which provides users with coding problems and evaluates them against test cases. The solutions to the problems can be in multiple programming languages and the traffic on the website can vary from time to time. When creating such a web-service, which deployment strategy do you use on-site servers, Iaas, Paas? and why? | 5 | 1 | Service Models | IaaS / PaaS Decision | Case Study | Medium |
| 1c | Consider a web application for building a BookShop using a web application. Your shop should support: Retrieve book details like price, recommendations. Adding a book to a shopping cart. Like a book. Sending recommendation of a book to a friend. Deleting a recommendation for a book. List out the resources for your application and the representations if you would like to see the data in JSON format. What should be the Accept syntax | 5 | 1 | Web Services & REST | REST Resource Design / JSON | Case Study | Hard |
| 1d | What are the steps associated to break a Monolith into Microservices? What difficulties are associated with this? | 5 | 1 | Microservices | Migration from Monolithic | Explain | Medium |
| 2a | What is Virtualization? Explain the pros and cons of Virtualization. | 5 | 2 | Virtualization | Pros & Cons | Explain | Easy |
| 2b | Mark as True or False: 1. The x86 architecture originally contained instructions that were nonvirtualizable using trap-and-emulate virtualization. 2. Shadow page tables store the mappings from guest physical to host physical memory addresses. 3. On a Type 2 VM, a software interface is created that emulates the devices with which a system would normally interact. 4. In the Kubernetes architecture, all these components are part of the Master node: API Server, etcd storage, scheduler and kubelet. 5. Docker container provides Hardware-level process isolation while Virtual Machine provides OS level process isolation. | 5 | 2 | Virtualization | Trap-and-Emulate, Shadow Page Tables, Type 2 VM, Kubernetes, Docker | Theory | Easy |
| 2c | Explain any 3 advantages and any 2 disadvantage of UnionFS | 5 | 2 | Docker UnionFS | Advantages & Disadvantages | Explain | Medium |
| 2d | Explain why DevOps is Needed? How is DevOps different from traditional software development and Operations process? | 5 | 2 | DevOps | DevOps vs Traditional | Comparison | Medium |
| 3a | What is storage virtualization? Explain categories of storage virtualization. | 5 | 3 | Block & Object Storage | Storage Categories | Explain | Medium |
| 3b | What is rebalancing of partitions? Explain various approaches for rebalancing partitions | 5 | 3 | Partitioning | Rebalancing Approaches | Explain | Medium |
| 3c | Explain Leader based replication technique | 5 | 3 | Replication | Leader-based Replication | Explain | Medium |
| 3d | Explain "Linearizability" in data replication | 5 | 3 | Consistency Models | Linearizability | Explain | Hard |
| 4a | What is a "fault-tolerant" system? If a service was unavailable for 20 minutes in 24 hours due to 2 failures, what is the uptime, MTBF and MTTR of that service? | 5 | 4 | Fault Tolerance | MTBF / MTTR Calculation | Numerical | Medium |
| 4b | Explain Failover Architecture. Hint: Active/Active and Active/Passive failover | 5 | 4 | Fault Tolerance | Failover Architecture | Explain | Medium |
| 4c | Explain modified ring "leader election" algorithm | 5 | 4 | Leader Election | Modified Ring Algorithm | Explain | Medium |
| 4d | What is Apache Zookeeper? How does it work? | 5 | 4 | Zookeeper | Working Mechanism | Explain | Medium |
| 5a | Explain the terms mentioned below from a cloud security perspective: 1. Cloud Time Service 2. Identity Management 3. Access Management 4. Break Glass Procedures 5. Key Management | 10 | 4 | Security | IAM / Key Management | Short Note | Medium |
| 5b | Explain the following Keystone concepts: 1. Roles 2. Assignment 3. Targets 4. Tokens 5. Catalog | 10 | 4 | Keystone/IAM | Keystone Components | Short Note | Medium |

---

### ─── 2022 (May 2022) ───

| Q.No | Exact Question | Marks | Unit | Main Topic | Subtopic | Type | Difficulty |
|------|----------------|-------|------|------------|----------|------|------------|
| 1a | If you are required to build a Cloud-Ready Application, how will you go about designing and building a cloud application architecture for private or public clouds? Explain 4 key steps. | 8 | 1 | Cloud Architecture | Cloud-Ready App Design | Explain | Medium |
| 1b | Describe with the help of examples various service models and deployment models of cloud computing. | 7 | 1 | Service & Deployment Models | IaaS/PaaS/SaaS + Deployment | Explain | Easy |
| 1c | Three applications are developed on the cloud – App1 is accessed using a browser on the cloud, App2 is installed on virtual machine and App3 is built using a cloud based database service. Classify the three apps into IaaS, PaaS and SaaS models with proper justification. Also, give a real life example of IaaS, PaaS and SaaS platform. | 5 | 1 | Service Models | IaaS/PaaS/SaaS Classification | Case Study | Medium |
| 2a | Consider a situation where you are required to apply any one of these types of virtualization — Full Virtualization, Bare Metal virtualization, Host based virtualization and Para Virtualization — to different implementation technologies. Mark the appropriate virtualization type for each requirement and justify your answer. (i) Run some dedicated applications on the VMs created on the guest OS and run some other applications on the host OS directly. (ii) Run special APIs requiring substantial OS modifications in a VM. (iii) Run non-critical instructions on the hardware directly while critical instructions are discovered and replaced with traps into the VMM to be emulated by software. (iv) Install the virtualization software directly on the hardware. | 8 | 2 | Virtualization Types | Full / Para / Bare Metal / Host-based | Case Study | Hard |
| 2b | Explain what are rings and what do the arrows in the following image represent? | 5 | 2 | Para Virtualization | Privilege Rings / VMM Diagram | Diagram | Medium |
| 2c | What are controller-manager, kubelets and pods in Kubernetes? Explain with a diagram where each of them execute – on master or worker? | 5 | 2 | Kubernetes | Kubernetes Architecture | Diagram | Medium |
| 2d | List one similarity and one difference between Docker container and a VM. | 2 | 2 | Containers | Docker vs VM | Comparison | Easy |
| 3a | (i) Explain Gluster file system architecture with a neat diagram (ii) How does Gluster file system compare with Lustre file system in terms of metadata management? | 10 | 3 | Object Stores | Gluster + Lustre Comparison | Diagram | Hard |
| 3b | Discuss 3 desirable properties of the CAP theorem and some of its practical implications. | 5 | 3 | CAP Theorem | Consistency, Availability, Partition Tolerance | Theory | Medium |
| 3c | What is a consistency model? Explain briefly any 4 types of consistency models. | 5 | 3 | Consistency Models | 4 Consistency Models | Explain | Medium |
| 4a | What is the purpose of Leader Election in Distributed computing? (2) Explain briefly Bully Algorithm and Leader election in a Ring (6) | 8 | 4 | Leader Election | Bully + Ring Algorithms | Explain | Medium |
| 4b | What is the problem with the implementation of a distributed lock in the following diagram? Explain with a diagram the approach that is used to overcome the problem. | 6 | 4 | Distributed Locking | Lock Lease Problem | Diagram | Hard |
| 4c | How does Zookeeper work? (3) What are the common services offered by Zookeeper? (3) | 6 | 4 | Zookeeper | Working + Services | Explain | Medium |
| 5a | Explain the following terms from Cloud Threat and Security Context. (2 Marks each): 1) Domain in Keystone 2) Token In Keystone 3) DoS Attack 4) Honeypot design pattern | 8 | 4 | Security / Keystone | Domain, Token, DoS, Honeypot | Short Note | Medium |
| 5b | What is Cloud Bursting? Explain how Cloud Bursting can be Beneficial to Cloud Users. | 6 | 4 | Cloud Bursting | Benefits of Cloud Bursting | Explain | Easy |
| 5c | What is multi-tenancy and mention its benefits in Cloud Computing. (2 Marks) You are asked to design a multitenant database for two universities – HighTechUniv and GlobalUniv to store information about students. HighTechUniv wants to store USN, student names and email ids while GlobalUniv wants to store USN, student names and grades. Design a multitenant database using the preallocated column method for the same. (4 Marks) | 6 | 4 | Multitenancy | Multitenant Database Design | Case Study | Hard |

---

### ─── 2023 July ───

| Q.No | Exact Question | Marks | Unit | Main Topic | Subtopic | Type | Difficulty |
|------|----------------|-------|------|------------|----------|------|------------|
| 1a | Explain the following terms with respect to Cloud Computing: 1. On-demand service 2. Rapid provisioning 3. Measured Service 4. Resource Pooling 5. Availability 6. Broad Network Access | 6 | 1 | Cloud Characteristics | NIST Properties | Definition | Easy |
| 1b | What is elasticity? How is it different from scalability? Give an example each to clearly distinguish these concepts. | 4 | 1 | Business Drivers / Scalability | Elasticity vs Scalability | Comparison | Easy |
| 1c | What are private, public, and hybrid clouds? Explain each of them. Discuss the relative advantages and disadvantages. | 10 | 1 | Deployment Models | Private/Public/Hybrid | Explain | Medium |
| 2a | Distinguish between bare metal and hosted hypervisors. Give an example of each. | 4 | 2 | Hypervisor Types | Type 1 vs Type 2 | Comparison | Easy |
| 2b | What is DevOps? What are the benefits derived from DevOps? How is DevOps different from traditional software development and Operations processes? | 4 | 2 | DevOps | DevOps vs Traditional | Comparison | Easy |
| 2c | Explain what are rings. Explain the instruction executions depicted in the following image. | 8 | 2 | Para Virtualization | Privilege Rings / VMM Diagram | Diagram | Medium |
| 2d | List at least four similarities and differences between Containers and VMs. | 4 | 2 | Containers | Containers vs VMs | Comparison | Easy |
| 3a | Explain what Gluster and Lustre are. Explain the Gluster architecture. How is it different from Lustre? | 4 | 3 | Object Stores | Gluster vs Lustre | Comparison | Medium |
| 3b | What is a consistency model? Explain Strict, Sequential, Causal and PRAM consistency. | 5 | 3 | Consistency Models | 4 Named Consistency Models | Explain | Medium |
| 3c | State CAP theorem. Discuss 3 desirable properties of the CAP theorem and some of its practical implications. | 9 | 3 | CAP Theorem | CAP + Practical Implications | Theory | Medium |
| 4a | What is the purpose of Leader Election in Distributed computing? Explain briefly Bully Algorithm and Modified Ring election. | 10 | 4 | Leader Election | Bully + Modified Ring | Explain | Medium |
| 4b | Explain how Zookeeper works. Explain the key benefits and the common services offered by Zookeeper. | 10 | 4 | Zookeeper | Working + Benefits + Services | Explain | Medium |
| 5a | Explain the terms mentioned below from a cloud security perspective: 1. Cloud Time Service 2. Identity Management 3. Access Management 4. Break Glass Procedures 5. Key Management | 10 | 4 | Security | IAM / Key Management | Short Note | Medium |
| 5b | What is multi-tenancy? Mention its benefits in Cloud Computing. You are asked to design a multitenant database for two universities – HighTechUniv and GlobalUniv to store information about students. HighTechUniv wants to store USN, student names and email ids while GlobalUniv wants to store USN, student names, and grades. Design a multitenant database using the preallocated column method for the same. | 6 | 4 | Multitenancy | Multitenant DB Design | Case Study | Hard |
| 5c | Explain the following security design patterns: a. Defense in Depth b. Honeypots | 4 | 4 | Security Design Patterns | Defense in Depth / Honeypots | Explain | Easy |

---

### ─── 2023 December ───

| Q.No | Exact Question | Marks | Unit | Main Topic | Subtopic | Type | Difficulty |
|------|----------------|-------|------|------------|----------|------|------------|
| 1a | Explain any 4 key design considerations that is used to build a cloud-ready application for private or public clouds (6M). Also, explain how Elasticity plays an important role in cloud computing? (2M) | 8 | 1 | Cloud Architecture / Elasticity | Cloud-Ready App + Elasticity | Explain | Medium |
| 1b | Compare and contrast private and public clouds clearly describing the advantages and disadvantages provided by each one of them. | 6 | 1 | Deployment Models | Private vs Public Cloud | Comparison | Easy |
| 1c | Explain briefly Bit Level parallelism, Instruction Level parallelism and Task Level parallelism. | 6 | 1 | Parallel Computing | Types of Parallelism | Explain | Medium |
| 2a | What is the difference between a hosted hypervisor and a bare metal hypervisor. Give an example of each (4M). List any four differences between a VM and a Container (4M) | 8 | 2 | Hypervisor Types / Containers | Type 1 vs Type 2 + VM vs Container | Comparison | Easy |
| 2b | Explain kubernetes architecture with a neat diagram clearly showing all the key components of both master and worker nodes. | 6 | 2 | Kubernetes | Kubernetes Architecture Diagram | Diagram | Medium |
| 2c | What is the difference between hot migration and cold migration? Explain pre-copy and post-copy techniques of hot migration. | 6 | 2 | VM Migration | Pre-copy / Post-copy | Explain | Medium |
| 3a | Explain Gluster file system architecture with a neat diagram. | 8 | 3 | Object Stores | Gluster Architecture | Diagram | Medium |
| 3b | Discuss 3 important properties of the CAP theorem and some of its practical implications while choosing a database for an application based on CAP theorem. | 6 | 3 | CAP Theorem | CAP Properties + DB Implications | Theory | Medium |
| 3c | What is the purpose of rebalancing of partitions? (2M) Explain Dynamic partitioning and Partitioning proportionally to the nodes (4M) | 6 | 3 | Partitioning | Dynamic + Proportional Partitioning | Explain | Medium |
| 4a | Explain Ring Election Algorithm with neat sketches. Clearly state the worst-case scenario and messages required in worst case scenario. (6M) What are the changes made in Modified Ring Election Algorithm to address the problem in Ring Election Algorithm? (2M) | 8 | 4 | Leader Election | Ring + Modified Ring | Diagram | Hard |
| 4b | What is the problem with the implementation of a distributed lock in the following diagram? (2M). What is a "fault-tolerant" system? Name at least two types of failures. (2M). If a service was unavailable for 60 minutes in 75 hours due to 6 failures, compute the MTBF and MTTR of that service? (2M) | 6 | 4 | Dist. Locking + Fault Tolerance | Lock Lease + MTBF/MTTR | Numerical | Medium |
| 4c | What is Zookeeper? How does it work? Name at least two contexts where zookeeper services may be used. | 6 | 4 | Zookeeper | Working + Use Cases | Explain | Easy |
| 5a | Explain the following terms used in Cloud Security: Domain in Keystone, Defense in Depth, Honeypot Design Pattern, Network Pattern | 8 | 4 | Security / Keystone | Keystone Domain + Patterns | Short Note | Medium |
| 5b | What is a DoS attack? Explain with appropriate sketches, how is DoS different from DDoS? Distinguish EDoS from the above. (1M+3M+2M) | 6 | 4 | Cloud Threats | DoS / DDoS / EDoS | Explain | Medium |
| 5c | What is a reverse proxy and what are its benefits? How is it different from a forward proxy? Provide a few applications where both are used. (2M + 2M + 2M) | 6 | 4 | Reverse Proxies | Forward vs Reverse Proxy | Comparison | Medium |

---

### ─── 2024 (Jan–May 2024) ───

> **Note:** In this paper, question numbering does not align strictly with unit topics. Q3a covers Microservices (Unit 1 topic), Q3b covers VM Migration (Unit 2 topic), Q3c covers Multitenancy (Unit 4 topic), and Q4a covers Partitioning/Replication (Unit 3 topic). Questions are listed by their paper number; topics are cross-referenced in unit-wise sections.

| Q.No | Exact Question | Marks | Unit (by #) | Main Topic | Subtopic | Type | Difficulty |
|------|----------------|-------|-------------|------------|----------|------|------------|
| 1a | How is scalability different from elasticity in the cloud? What type of scalability is present in cloud? Discuss the type that you chose. "Private cloud do not need to address the challenge of elasticity" - state whether this statement is true/false and justify the statement. | 10 | 1 | Scalability / Elasticity | Horizontal Scalability + Private Cloud | Explain | Medium |
| 1b | Discuss the relative advantages and disadvantages of private and public clouds. Mention the tools/technologies that illustrate private and public cloud features or examples | 10 | 1 | Deployment Models | Private vs Public Cloud | Comparison | Medium |
| 1c | Discuss the three cloud service models with suitable examples. | 5 | 1 | Service Models | IaaS / PaaS / SaaS | Explain | Easy |
| 2a | Discuss different types of hypervisors. Give an example of each. How paravirtualization and full virtualization techniques are used to virtualize x86. | 10 | 2 | Hypervisor Types | Type 1/2 + Para/Full Virt | Explain | Medium |
| 2b | Consider an architecture which supports the following instructions: (i) mark each of the instructions as to whether they are sensitive and if so whether they are behaviour or control sensitive (6). (ii) based on this data will you be able to design a trap-and-emulate hypervisor for this architecture. Justify your solution | 10 | 2 | Goldberg-Popek / Trap & Emulate | Sensitive Instructions Classification | Numerical/Theory | Hard |
| 2c | How do you differentiate between container and Virtual Machines. Mention example tools/technologies for containers and virtual machines. | 5 | 2 | Containers | Container vs VM | Comparison | Easy |
| 3a | What is a microservice application and how is it different from a regular monolithic application? How is REST related to the microservice programming model? Discuss REST in brief. | 10 | 3* | Microservices / REST | Monolithic vs Microservices + REST | Explain | Medium |
| 3b | Bring out the comparison between hot migration and cold migration? Explain the different copy techniques of hot migration. | 10 | 3* | VM Migration | Hot vs Cold + Pre/Post Copy | Comparison | Medium |
| 3c | You are asked to design a multitenant database for two hospitals – BestCare and PatientFriendly to store information about patients. BestCare wants to store PatientID, names and previous history while PatientFriendly wants to store PatientID, names and NextAppointment date. Design a multitenant database using the preallocated column method for the same. | 5 | 3* | Multitenancy | Preallocated Column Method | Case Study | Hard |
| 4a | Discuss the rebalancing of partitions in cloud storage. Bring out any of the hash based partitioning methods. Discuss the leaderless replication method to keep copies of data in cloud storage. | 10 | 4* | Partitioning + Replication | Rebalancing + Hash-based + Leaderless | Explain | Hard |
| 4b | Which are the three major leader election algorithms in cloud systems. Discuss the relative merits and demerits of Ring algorithm along with details on message latencies. | 10 | 4 | Leader Election | Ring, Modified Ring, Bully — Merits/Demerits | Explain | Hard |
| 4c | What is a reverse proxy? What additional features can it provide. | 5 | 4 | Reverse Proxies | Features of Reverse Proxy | Explain | Easy |

*Topics placed under question number unit; actual syllabus unit mapping differs.

---

### ─── 2025 (Dec 2025) ───

| Q.No | Exact Question | Marks | Unit | Main Topic | Subtopic | Type | Difficulty |
|------|----------------|-------|------|------------|----------|------|------------|
| 1a | What are IaaS, PaaS, SaaS with respect to cloud computing? Illustrate with suitable examples. Discuss the relative advantages and disadvantages of these | 8 | 1 | Service Models | IaaS/PaaS/SaaS Advantages/Disadvantages | Explain | Easy |
| 1b | Explain Web Services and RESTful Architecture in Cloud Applications. Explain any three mandatory constraints (principles) for designing a RESTful system | 8 | 1 | Web Services & REST | RESTful Constraints/Principles | Explain | Medium |
| 1c | Describe the Message Queue based Communication Model and the Publish–Subscribe Pattern. | 4 | 1 | Message Queues / Pub-Sub | MQ + Pub-Sub Model | Explain | Easy |
| 1d | There are well defined strategies available to migrate an application to cloud. Among these, explain any 5 methods in short. Methods are - Rehost, Re-platform, Re-architecting, Re-purchase, Retire, Retain | 5 | 1 | Migration Strategies | 5R / 6R Cloud Migration | Explain | Easy |
| 2a | Why virtualization is required for cloud computing? What are type 1 and type 2 virtualizations? | 5 | 2 | Hypervisor Types | Type 1 vs Type 2 | Explain | Easy |
| 2b | What are shadow page table and extended page tables with respect to virtualization? Discuss the working features of each of these techniques. | 8 | 2 | Shadow & Nested Page Tables | Shadow PT vs Extended PT | Explain | Hard |
| 2c | What are the major modules in Docker architecture? Discuss their role in brief. Mention any three commands of Docker with their working | 6 | 2 | Docker | Docker Architecture Modules | Explain | Medium |
| 2d | What are the advantages and disadvantages of pre-copy and post-copy migration methods? | 3 | 2 | VM Migration | Pre-copy vs Post-copy | Comparison | Medium |
| 3a | While data partitioning is applied for the data stored in cloud storage, there are situations that lead to repartitioning. List those situations. Once repartitioning is applied, what are the minimum requirements that needs to be satisfied. | 5 | 3 | Partitioning | Repartitioning Triggers & Requirements | Explain | Medium |
| 3b | In leader-based data replication, different types of replicas are maintained. Explain why these types of replication approaches are required. Additionally, discuss the factors that influence the decision regarding the number of replicas to be maintained. | 8 | 3 | Replication | Leader-based + Replica Factors | Explain | Medium |
| 3c | Discuss CAP theorem. Explain the working of two-phase commit protocol as compared to transactions in RDBMS. | 8 | 3 | CAP Theorem / Transactions | CAP + Two-Phase Commit | Explain | Hard |
| 3d | With reference to the consistency model, why linearizability is important? How Compare and Set works in this context? | 4 | 3 | Consistency Models | Linearizability + CAS | Explain | Hard |
| 4a | Discuss Ring, Modified Ring and Bully algorithm with suitable examples. How they are different from each other | 12 | 4 | Leader Election | All 3 Algorithms Compared | Explain | Hard |
| 4b | In the context of content delivery, load balancing there are proxy servers configured in the web application environment. How these proxy servers work and illustrate the working of the 2 major varieties of proxy servers, with their relative merits and demerits. | 8 | 4 | Reverse Proxies | Forward + Reverse Proxy Working | Explain | Medium |
| 4c | Consider the given diagram - what are domains, groups, users and projects, Roles in the context of Openstack Keystone. | 5 | 4 | Keystone/IAM | Keystone Concepts from Diagram | Diagram | Medium |

---

---

## STEP 2 — UNIT-WISE ORGANIZATION

---

# UNIT 1 — Cloud Programming Models

---

## Topic: Service Models (IaaS / PaaS / SaaS)

| Year | Q.No | Exact Question | Marks | Frequency |
|------|------|----------------|-------|-----------|
| 2021 | 1b | Imagine if you are developing the back-end service for a website like codechef which provides users with coding problems and evaluates them against test cases. The solutions to the problems can be in multiple programming languages and the traffic on the website can vary from time to time. When creating such a web-service, which deployment strategy do you use on-site servers, Iaas, Paas? and why? | 5 | 1 |
| 2022 | 1b | Describe with the help of examples various service models and deployment models of cloud computing. | 7 | 1 |
| 2022 | 1c | Three applications are developed on the cloud – App1 is accessed using a browser on the cloud, App2 is installed on virtual machine and App3 is built using a cloud based database service. Classify the three apps into IaaS, PaaS and SaaS models with proper justification. Also, give a real life example of IaaS, PaaS and SaaS platform. | 5 | 1 |
| 2024 | 1c | Discuss the three cloud service models with suitable examples. | 5 | 1 |
| 2025 | 1a | What are IaaS, PaaS, SaaS with respect to cloud computing? Illustrate with suitable examples. Discuss the relative advantages and disadvantages of these | 8 | 1 |

**Total Appearances: 5 | Total Marks: 30**

---

## Topic: Deployment Models (Private / Public / Hybrid)

| Year | Q.No | Exact Question | Marks | Frequency |
|------|------|----------------|-------|-----------|
| 2021 | 1a | The International Data Corporation has found that 80% of organisations have moved applications out of public cloud and back to private cloud. Why are companies moving certain applications and workloads back to private cloud? What are the issues you might encounter if you go "all in" with public cloud? | 5 | 1 |
| 2022 | 1b | Describe with the help of examples various service models and deployment models of cloud computing. | 7 | 1 |
| 2023 Jul | 1c | What are private, public, and hybrid clouds? Explain each of them. Discuss the relative advantages and disadvantages. | 10 | 1 |
| 2023 Dec | 1b | Compare and contrast private and public clouds clearly describing the advantages and disadvantages provided by each one of them. | 6 | 1 |
| 2024 | 1b | Discuss the relative advantages and disadvantages of private and public clouds. Mention the tools/technologies that illustrate private and public cloud features or examples | 10 | 1 |

**Total Appearances: 5 | Total Marks: 38**

---

## Topic: Cloud Architecture / Cloud-Ready Application Design

| Year | Q.No | Exact Question | Marks | Frequency |
|------|------|----------------|-------|-----------|
| 2022 | 1a | If you are required to build a Cloud-Ready Application, how will you go about designing and building a cloud application architecture for private or public clouds? Explain 4 key steps. | 8 | 1 |
| 2023 Dec | 1a | Explain any 4 key design considerations that is used to build a cloud-ready application for private or public clouds (6M). Also, explain how Elasticity plays an important role in cloud computing? (2M) | 8 | 1 |

**Total Appearances: 2 | Total Marks: 16**

---

## Topic: Scalability / Elasticity

| Year | Q.No | Exact Question | Marks | Frequency |
|------|------|----------------|-------|-----------|
| 2023 Jul | 1b | What is elasticity? How is it different from scalability? Give an example each to clearly distinguish these concepts. | 4 | 1 |
| 2023 Dec | 1a | (part) Also, explain how Elasticity plays an important role in cloud computing? | 2 | 1 |
| 2024 | 1a | How is scalability different from elasticity in the cloud? What type of scalability is present in cloud? Discuss the type that you chose. "Private cloud do not need to address the challenge of elasticity" - state whether this statement is true/false and justify the statement. | 10 | 1 |

**Total Appearances: 3 | Total Marks: 16**

---

## Topic: Web Services and REST

| Year | Q.No | Exact Question | Marks | Frequency |
|------|------|----------------|-------|-----------|
| 2021 | 1c | Consider a web application for building a BookShop using a web application... List out the resources for your application and the representations if you would like to see the data in JSON format. What should be the Accept syntax | 5 | 1 |
| 2024 | 3a | (part) How is REST related to the microservice programming model? Discuss REST in brief. | (part of 10) | 1 |
| 2025 | 1b | Explain Web Services and RESTful Architecture in Cloud Applications. Explain any three mandatory constraints (principles) for designing a RESTful system | 8 | 1 |

**Total Appearances: 3 | Total Marks: ~21**

---

## Topic: Microservices / Monolithic vs Microservices

| Year | Q.No | Exact Question | Marks | Frequency |
|------|------|----------------|-------|-----------|
| 2021 | 1d | What are the steps associated to break a Monolith into Microservices? What difficulties are associated with this? | 5 | 1 |
| 2024 | 3a | What is a microservice application and how is it different from a regular monolithic application? How is REST related to the microservice programming model? Discuss REST in brief. | 10 | 1 |

**Total Appearances: 2 | Total Marks: 15**

---

## Topic: Cloud Characteristics (NIST Properties)

| Year | Q.No | Exact Question | Marks | Frequency |
|------|------|----------------|-------|-----------|
| 2023 Jul | 1a | Explain the following terms with respect to Cloud Computing: 1. On-demand service 2. Rapid provisioning 3. Measured Service 4. Resource Pooling 5. Availability 6. Broad Network Access | 6 | 1 |

**Total Appearances: 1 | Total Marks: 6**

---

## Topic: Parallel Computing

| Year | Q.No | Exact Question | Marks | Frequency |
|------|------|----------------|-------|-----------|
| 2023 Dec | 1c | Explain briefly Bit Level parallelism, Instruction Level parallelism and Task Level parallelism. | 6 | 1 |

**Total Appearances: 1 | Total Marks: 6**

---

## Topic: Message Queues / Pub-Sub Model

| Year | Q.No | Exact Question | Marks | Frequency |
|------|------|----------------|-------|-----------|
| 2025 | 1c | Describe the Message Queue based Communication Model and the Publish–Subscribe Pattern. | 4 | 1 |

**Total Appearances: 1 | Total Marks: 4**

---

## Topic: Cloud Migration Strategies (5Rs / 6Rs)

| Year | Q.No | Exact Question | Marks | Frequency |
|------|------|----------------|-------|-----------|
| 2025 | 1d | There are well defined strategies available to migrate an application to cloud. Among these, explain any 5 methods in short. Methods are - Rehost, Re-platform, Re-architecting, Re-purchase, Retire, Retain | 5 | 1 |

**Total Appearances: 1 | Total Marks: 5**

---

---

# UNIT 2 — Virtualization

---

## Topic: Hypervisor Types (Type 1 / Type 2 / Bare Metal / Hosted)

| Year | Q.No | Exact Question | Marks | Frequency |
|------|------|----------------|-------|-----------|
| 2021 | 2a | What is Virtualization? Explain the pros and cons of Virtualization. | 5 | 1 |
| 2022 | 2a | Consider a situation where you are required to apply any one of these types of virtualization: Full Virtualization, Bare Metal virtualization, Host based virtualization and Para Virtualization... | 8 | 1 |
| 2023 Jul | 2a | Distinguish between bare metal and hosted hypervisors. Give an example of each. | 4 | 1 |
| 2023 Dec | 2a | What is the difference between a hosted hypervisor and a bare metal hypervisor. Give an example of each (4M). List any four differences between a VM and a Container (4M) | 8 | 1 |
| 2024 | 2a | Discuss different types of hypervisors. Give an example of each. How paravirtualization and full virtualization techniques are used to virtualize x86. | 10 | 1 |
| 2025 | 2a | Why virtualization is required for cloud computing? What are type 1 and type 2 virtualizations? | 5 | 1 |

**Total Appearances: 6 | Total Marks: 40**

---

## Topic: Para Virtualization / Privilege Rings / VMM Diagram

| Year | Q.No | Exact Question | Marks | Frequency |
|------|------|----------------|-------|-----------|
| 2022 | 2b | Explain what are rings and what do the arrows in the following image represent? | 5 | 1 |
| 2023 Jul | 2c | Explain what are rings. Explain the instruction executions depicted in the following image. | 8 | 1 |

**Total Appearances: 2 | Total Marks: 13**

---

## Topic: Trap and Emulate / Goldberg-Popek Principles

| Year | Q.No | Exact Question | Marks | Frequency |
|------|------|----------------|-------|-----------|
| 2021 | 2b | (True/False) 1. The x86 architecture originally contained instructions that were nonvirtualizable using trap-and-emulate virtualization | 1 (of 5) | 1 |
| 2024 | 2b | Consider an architecture which supports the following instructions: (i) mark each of the instructions as to whether they are sensitive and if so whether they are behaviour or control sensitive (6). (ii) based on this data will you be able to design a trap-and-emulate hypervisor for this architecture. Justify your solution | 10 | 1 |

**Total Appearances: 2 | Total Marks: ~11**

---

## Topic: Shadow Page Tables / Nested Page Tables

| Year | Q.No | Exact Question | Marks | Frequency |
|------|------|----------------|-------|-----------|
| 2021 | 2b | (True/False) 2. Shadow page tables store the mappings from guest physical to host physical memory addresses. | 1 (of 5) | 1 |
| 2025 | 2b | What are shadow page table and extended page tables with respect to virtualization? Discuss the working features of each of these techniques. | 8 | 1 |

**Total Appearances: 2 | Total Marks: ~9**

---

## Topic: Kubernetes Orchestration

| Year | Q.No | Exact Question | Marks | Frequency |
|------|------|----------------|-------|-----------|
| 2021 | 2b | (True/False) 4. In the Kubernetes architecture, all these components are part of the Master node: API Server, etcd storage, scheduler and kubelet | 1 (of 5) | 1 |
| 2022 | 2c | What are controller-manager, kubelets and pods in Kubernetes? Explain with a diagram where each of them execute – on master or worker? | 5 | 1 |
| 2023 Dec | 2b | Explain kubernetes architecture with a neat diagram clearly showing all the key components of both master and worker nodes. | 6 | 1 |

**Total Appearances: 3 | Total Marks: ~12**

---

## Topic: Docker (UnionFS / Architecture / Commands)

| Year | Q.No | Exact Question | Marks | Frequency |
|------|------|----------------|-------|-----------|
| 2021 | 2c | Explain any 3 advantages and any 2 disadvantage of UnionFS | 5 | 1 |
| 2025 | 2c | What are the major modules in Docker architecture? Discuss their role in brief. Mention any three commands of Docker with their working | 6 | 1 |

**Total Appearances: 2 | Total Marks: 11**

---

## Topic: Containers vs VMs

| Year | Q.No | Exact Question | Marks | Frequency |
|------|------|----------------|-------|-----------|
| 2021 | 2b | (True/False) 5. Docker container provides Hardware-level process isolation while Virtual Machine provides OS level process isolation | 1 (of 5) | 1 |
| 2022 | 2d | List one similarity and one difference between Docker container and a VM. | 2 | 1 |
| 2023 Jul | 2d | List at least four similarities and differences between Containers and VMs. | 4 | 1 |
| 2023 Dec | 2a | List any four differences between a VM and a Container (4M) | 4 | 1 |
| 2024 | 2c | How do you differentiate between container and Virtual Machines. Mention example tools/technologies for containers and virtual machines. | 5 | 1 |

**Total Appearances: 5 | Total Marks: ~16**

---

## Topic: DevOps

| Year | Q.No | Exact Question | Marks | Frequency |
|------|------|----------------|-------|-----------|
| 2021 | 2d | Explain why DevOps is Needed? How is DevOps different from traditional software development and Operations process? | 5 | 1 |
| 2023 Jul | 2b | What is DevOps? What are the benefits derived from DevOps? How is DevOps different from traditional software development and Operations processes? | 4 | 1 |

**Total Appearances: 2 | Total Marks: 9**

---

## Topic: VM Migration (Hot / Cold / Pre-copy / Post-copy)

| Year | Q.No | Exact Question | Marks | Frequency |
|------|------|----------------|-------|-----------|
| 2023 Dec | 2c | What is the difference between hot migration and cold migration? Explain pre-copy and post-copy techniques of hot migration. | 6 | 1 |
| 2024 | 3b | Bring out the comparison between hot migration and cold migration? Explain the different copy techniques of hot migration. | 10 | 1 |
| 2025 | 2d | What are the advantages and disadvantages of pre-copy and post-copy migration methods? | 3 | 1 |

**Total Appearances: 3 | Total Marks: 19**

---

---

# UNIT 3 — Distributed Storage

---

## Topic: Gluster / Lustre File System Architecture

| Year | Q.No | Exact Question | Marks | Frequency |
|------|------|----------------|-------|-----------|
| 2022 | 3a | (i) Explain Gluster file system architecture with a neat diagram (ii) How does Gluster file system compare with Lustre file system in terms of metadata management? | 10 | 1 |
| 2023 Jul | 3a | Explain what Gluster and Lustre are. Explain the Gluster architecture. How is it different from Lustre? | 4 | 1 |
| 2023 Dec | 3a | Explain Gluster file system architecture with a neat diagram. | 8 | 1 |

**Total Appearances: 3 | Total Marks: 22**

---

## Topic: CAP Theorem

| Year | Q.No | Exact Question | Marks | Frequency |
|------|------|----------------|-------|-----------|
| 2022 | 3b | Discuss 3 desirable properties of the CAP theorem and some of its practical implications. | 5 | 1 |
| 2023 Jul | 3c | State CAP theorem. Discuss 3 desirable properties of the CAP theorem and some of its practical implications. | 9 | 1 |
| 2023 Dec | 3b | Discuss 3 important properties of the CAP theorem and some of its practical implications while choosing a database for an application based on CAP theorem. | 6 | 1 |
| 2025 | 3c | Discuss CAP theorem. Explain the working of two-phase commit protocol as compared to transactions in RDBMS. | 8 | 1 |

**Total Appearances: 4 | Total Marks: 28**

---

## Topic: Consistency Models

| Year | Q.No | Exact Question | Marks | Frequency |
|------|------|----------------|-------|-----------|
| 2021 | 3d | Explain "Linearizability" in data replication | 5 | 1 |
| 2022 | 3c | What is a consistency model? Explain briefly any 4 types of consistency models. | 5 | 1 |
| 2023 Jul | 3b | What is a consistency model? Explain Strict, Sequential, Causal and PRAM consistency. | 5 | 1 |
| 2025 | 3d | With reference to the consistency model, why linearizability is important? How Compare and Set works in this context? | 4 | 1 |

**Total Appearances: 4 | Total Marks: 19**

---

## Topic: Rebalancing Partitions

| Year | Q.No | Exact Question | Marks | Frequency |
|------|------|----------------|-------|-----------|
| 2021 | 3b | What is rebalancing of partitions? Explain various approaches for rebalancing partitions | 5 | 1 |
| 2023 Dec | 3c | What is the purpose of rebalancing of partitions? (2M) Explain Dynamic partitioning and Partitioning proportionally to the nodes (4M) | 6 | 1 |
| 2024 | 4a | Discuss the rebalancing of partitions in cloud storage. Bring out any of the hash based partitioning methods. Discuss the leaderless replication method to keep copies of data in cloud storage. | 10 | 1 |
| 2025 | 3a | While data partitioning is applied for the data stored in cloud storage, there are situations that lead to repartitioning. List those situations. Once repartitioning is applied, what are the minimum requirements that needs to be satisfied. | 5 | 1 |

**Total Appearances: 4 | Total Marks: 26**

---

## Topic: Replication (Leader-based / Leaderless / Multi-leader)

| Year | Q.No | Exact Question | Marks | Frequency |
|------|------|----------------|-------|-----------|
| 2021 | 3c | Explain Leader based replication technique | 5 | 1 |
| 2024 | 4a | (part) Discuss the leaderless replication method to keep copies of data in cloud storage. | (part of 10) | 1 |
| 2025 | 3b | In leader-based data replication, different types of replicas are maintained. Explain why these types of replication approaches are required. Additionally, discuss the factors that influence the decision regarding the number of replicas to be maintained. | 8 | 1 |

**Total Appearances: 3 | Total Marks: ~21**

---

## Topic: Storage Virtualization

| Year | Q.No | Exact Question | Marks | Frequency |
|------|------|----------------|-------|-----------|
| 2021 | 3a | What is storage virtualization? Explain categories of storage virtualization. | 5 | 1 |

**Total Appearances: 1 | Total Marks: 5**

---

## Topic: Transactions / Two-Phase Commit

| Year | Q.No | Exact Question | Marks | Frequency |
|------|------|----------------|-------|-----------|
| 2025 | 3c | (part) Explain the working of two-phase commit protocol as compared to transactions in RDBMS. | (part of 8) | 1 |

**Total Appearances: 1 | Total Marks: ~4**

---

---

# UNIT 4 — Cloud Controller, Performance, Scalability and Security

---

## Topic: Leader Election Algorithms (Ring / Modified Ring / Bully)

| Year | Q.No | Exact Question | Marks | Frequency |
|------|------|----------------|-------|-----------|
| 2021 | 4c | Explain modified ring "leader election" algorithm | 5 | 1 |
| 2022 | 4a | What is the purpose of Leader Election in Distributed computing? (2) Explain briefly Bully Algorithm and Leader election in a Ring (6) | 8 | 1 |
| 2023 Jul | 4a | What is the purpose of Leader Election in Distributed computing? Explain briefly Bully Algorithm and Modified Ring election. | 10 | 1 |
| 2023 Dec | 4a | Explain Ring Election Algorithm with neat sketches. Clearly state the worst-case scenario and messages required in worst case scenario. (6M) What are the changes made in Modified Ring Election Algorithm to address the problem in Ring Election Algorithm? (2M) | 8 | 1 |
| 2024 | 4b | Which are the three major leader election algorithms in cloud systems. Discuss the relative merits and demerits of Ring algorithm along with details on message latencies. | 10 | 1 |
| 2025 | 4a | Discuss Ring, Modified Ring and Bully algorithm with suitable examples. How they are different from each other | 12 | 1 |

**Total Appearances: 6 | Total Marks: 53**

---

## Topic: Apache Zookeeper

| Year | Q.No | Exact Question | Marks | Frequency |
|------|------|----------------|-------|-----------|
| 2021 | 4d | What is Apache Zookeeper? How does it work? | 5 | 1 |
| 2022 | 4c | How does Zookeeper work? (3) What are the common services offered by Zookeeper? (3) | 6 | 1 |
| 2023 Jul | 4b | Explain how Zookeeper works. Explain the key benefits and the common services offered by Zookeeper. | 10 | 1 |
| 2023 Dec | 4c | What is Zookeeper? How does it work? Name at least two contexts where zookeeper services may be used. | 6 | 1 |

**Total Appearances: 4 | Total Marks: 27**

---

## Topic: Distributed Locking (Fencing Token / Lease Problem)

| Year | Q.No | Exact Question | Marks | Frequency |
|------|------|----------------|-------|-----------|
| 2022 | 4b | What is the problem with the implementation of a distributed lock in the following diagram? Explain with a diagram the approach that is used to overcome the problem. | 6 | 1 |
| 2023 Dec | 4b | What is the problem with the implementation of a distributed lock in the following diagram? (2M) | 2 | 1 |

**Total Appearances: 2 | Total Marks: 8**

---

## Topic: Fault Tolerance / MTBF / MTTR

| Year | Q.No | Exact Question | Marks | Frequency |
|------|------|----------------|-------|-----------|
| 2021 | 4a | What is a "fault-tolerant" system? If a service was unavailable for 20 minutes in 24 hours due to 2 failures, what is the uptime, MTBF and MTTR of that service? | 5 | 1 |
| 2023 Dec | 4b | What is a "fault-tolerant" system? Name at least two types of failures. (2M) If a service was unavailable for 60 minutes in 75 hours due to 6 failures, compute the MTBF and MTTR of that service? (2M) | 4 | 1 |

**Total Appearances: 2 | Total Marks: 9**

---

## Topic: Failover Architecture

| Year | Q.No | Exact Question | Marks | Frequency |
|------|------|----------------|-------|-----------|
| 2021 | 4b | Explain Failover Architecture. Hint: Active/Active and Active/Passive failover | 5 | 1 |

**Total Appearances: 1 | Total Marks: 5**

---

## Topic: Cloud Security (IAM / Identity / Access / Break Glass / Key Management)

| Year | Q.No | Exact Question | Marks | Frequency |
|------|------|----------------|-------|-----------|
| 2021 | 5a | Explain the terms mentioned below from a cloud security perspective: 1. Cloud Time Service 2. Identity Management 3. Access Management 4. Break Glass Procedures 5. Key Management | 10 | 1 |
| 2023 Jul | 5a | Explain the terms mentioned below from a cloud security perspective: 1. Cloud Time Service 2. Identity Management 3. Access Management 4. Break Glass Procedures 5. Key Management | 10 | 1 |

**Total Appearances: 2 | Total Marks: 20 (IDENTICAL QUESTION REPEATED)**

---

## Topic: Keystone / IAM (OpenStack)

| Year | Q.No | Exact Question | Marks | Frequency |
|------|------|----------------|-------|-----------|
| 2021 | 5b | Explain the following Keystone concepts: 1. Roles 2. Assignment 3. Targets 4. Tokens 5. Catalog | 10 | 1 |
| 2022 | 5a | Explain the following terms from Cloud Threat and Security Context: 1) Domain in Keystone 2) Token In Keystone | 4 (of 8) | 1 |
| 2023 Dec | 5a | (part) Domain in Keystone | 2 (of 8) | 1 |
| 2025 | 4c | Consider the given diagram - what are domains, groups, users and projects, Roles in the context of Openstack Keystone. | 5 | 1 |

**Total Appearances: 4 | Total Marks: ~21**

---

## Topic: Cloud Threats / DoS / DDoS / EDoS

| Year | Q.No | Exact Question | Marks | Frequency |
|------|------|----------------|-------|-----------|
| 2022 | 5a | (part) 3) DoS Attack | 2 (of 8) | 1 |
| 2023 Dec | 5b | What is a DoS attack? Explain with appropriate sketches, how is DoS different from DDoS? Distinguish EDoS from the above. (1M+3M+2M) | 6 | 1 |

**Total Appearances: 2 | Total Marks: 8**

---

## Topic: Security Design Patterns (Defense in Depth / Honeypot / Network)

| Year | Q.No | Exact Question | Marks | Frequency |
|------|------|----------------|-------|-----------|
| 2022 | 5a | (part) 4) Honeypot design pattern | 2 (of 8) | 1 |
| 2023 Jul | 5c | Explain the following security design patterns: a. Defense in Depth b. Honeypots | 4 | 1 |
| 2023 Dec | 5a | (part) Defense in Depth, Honeypot Design Pattern, Network Pattern | 6 (of 8) | 1 |

**Total Appearances: 3 | Total Marks: ~12**

---

## Topic: Multitenancy / Multitenant Database

| Year | Q.No | Exact Question | Marks | Frequency |
|------|------|----------------|-------|-----------|
| 2022 | 5c | What is multi-tenancy and mention its benefits in Cloud Computing. (2 Marks) You are asked to design a multitenant database for two universities – HighTechUniv and GlobalUniv... Design a multitenant database using the preallocated column method for the same. (4 Marks) | 6 | 1 |
| 2023 Jul | 5b | What is multi-tenancy? Mention its benefits in Cloud Computing. You are asked to design a multitenant database for two universities – HighTechUniv and GlobalUniv... Design a multitenant database using the preallocated column method for the same. | 6 | 1 |
| 2024 | 3c | You are asked to design a multitenant database for two hospitals – BestCare and PatientFriendly... Design a multitenant database using the preallocated column method for the same. | 5 | 1 |

**Total Appearances: 3 | Total Marks: 17 (Nearly identical question in 2022 & 2023 Jul)**

---

## Topic: Cloud Bursting

| Year | Q.No | Exact Question | Marks | Frequency |
|------|------|----------------|-------|-----------|
| 2022 | 5b | What is Cloud Bursting? Explain how Cloud Bursting can be Beneficial to Cloud Users. | 6 | 1 |

**Total Appearances: 1 | Total Marks: 6**

---

## Topic: Reverse Proxies / Forward Proxies

| Year | Q.No | Exact Question | Marks | Frequency |
|------|------|----------------|-------|-----------|
| 2023 Dec | 5c | What is a reverse proxy and what are its benefits? How is it different from a forward proxy? Provide a few applications where both are used. (2M + 2M + 2M) | 6 | 1 |
| 2024 | 4c | What is a reverse proxy? What additional features can it provide. | 5 | 1 |
| 2025 | 4b | In the context of content delivery, load balancing there are proxy servers configured in the web application environment. How these proxy servers work and illustrate the working of the 2 major varieties of proxy servers, with their relative merits and demerits. | 8 | 1 |

**Total Appearances: 3 | Total Marks: 19**

---

---

## STEP 3 — FREQUENCY ANALYSIS

### Most Repeated Topics (All Units Combined)

| Rank | Topic | Unit | Times Asked | Total Marks |
|------|-------|------|-------------|-------------|
| 1 | Leader Election Algorithms (Ring/Modified Ring/Bully) | 4 | 6 | 53 |
| 2 | Hypervisor Types (Type 1 / Type 2 / Bare Metal / Hosted) | 2 | 6 | 40 |
| 3 | Deployment Models (Private/Public/Hybrid Cloud) | 1 | 5 | 38 |
| 4 | Service Models (IaaS/PaaS/SaaS) | 1 | 5 | 30 |
| 5 | Containers vs VMs | 2 | 5 | ~16 |
| 6 | CAP Theorem | 3 | 4 | 28 |
| 7 | Consistency Models | 3 | 4 | 19 |
| 8 | Rebalancing Partitions | 3 | 4 | 26 |
| 9 | Zookeeper | 4 | 4 | 27 |
| 10 | Keystone/IAM | 4 | 4 | ~21 |
| 11 | Gluster/Lustre Architecture | 3 | 3 | 22 |
| 12 | VM Migration (Hot/Cold/Pre-copy/Post-copy) | 2 | 3 | 19 |
| 13 | Multitenancy / Multitenant DB | 4 | 3 | 17 |
| 14 | Security Design Patterns (Honeypot/Defense in Depth) | 4 | 3 | ~12 |
| 15 | Reverse Proxies | 4 | 3 | 19 |
| 16 | Cloud Architecture / Cloud-Ready App | 1 | 2 | 16 |
| 17 | Web Services & REST | 1 | 3 | ~21 |
| 18 | Scalability / Elasticity | 1 | 3 | 16 |
| 19 | Kubernetes Architecture | 2 | 3 | ~12 |
| 20 | Replication (Leader-based/Leaderless) | 3 | 3 | ~21 |
| 21 | DevOps | 2 | 2 | 9 |
| 22 | Para Virtualization / Privilege Rings Diagram | 2 | 2 | 13 |
| 23 | Fault Tolerance / MTBF / MTTR | 4 | 2 | 9 |
| 24 | Distributed Locking | 4 | 2 | 8 |
| 25 | Cloud Threats (DoS/DDoS/EDoS) | 4 | 2 | 8 |
| 26 | IAM Security Terms (Cloud Time Service etc.) | 4 | 2 | 20 |
| 27 | Docker Architecture/UnionFS | 2 | 2 | 11 |
| 28 | Trap and Emulate / Goldberg-Popek | 2 | 2 | ~11 |
| 29 | Microservices / Monolithic | 1 | 2 | 15 |

---

### Directly Repeated Questions (Same or Near-Identical Wording)

| # | Question Pattern | Years | Marks Each |
|---|------------------|-------|------------|
| 1 | **EXACT REPEAT:** "Explain the terms mentioned below from a cloud security perspective: 1. Cloud Time Service 2. Identity Management 3. Access Management 4. Break Glass Procedures 5. Key Management" | 2021, 2023 Jul | 10, 10 |
| 2 | **NEAR-EXACT REPEAT:** Multitenant DB design for two universities (HighTechUniv/GlobalUniv) using preallocated column method | 2022, 2023 Jul | 6, 6 |
| 3 | "Discuss 3 desirable properties of the CAP theorem and some of its practical implications" | 2022, 2023 Jul, 2023 Dec | 5, 9, 6 |
| 4 | Leader Election — asking about Ring + Modified Ring + Bully in combination | 2022, 2023 Jul, 2023 Dec, 2024, 2025 | 8, 10, 8, 10, 12 |
| 5 | Zookeeper — "How does it work + services offered" | 2021, 2022, 2023 Jul, 2023 Dec | 5, 6, 10, 6 |
| 6 | "Distributed lock diagram — what is the problem?" (same diagram) | 2022, 2023 Dec | 6, 2 |
| 7 | "Fault-tolerant system + MTBF/MTTR numerical" | 2021, 2023 Dec | 5, 4 |
| 8 | Private vs Public cloud advantages/disadvantages | 2022, 2023 Jul, 2023 Dec, 2024 | 7, 10, 6, 10 |
| 9 | IaaS/PaaS/SaaS with examples | 2022, 2024, 2025 | 7, 5, 8 |
| 10 | Gluster architecture with diagram | 2022, 2023 Jul, 2023 Dec | 10, 4, 8 |
| 11 | Hot vs Cold migration + pre-copy/post-copy | 2023 Dec, 2024, 2025 | 6, 10, 3 |
| 12 | Bare metal vs hosted hypervisor + example | 2022, 2023 Jul, 2023 Dec, 2024 | 8, 4, 8, 10 |
| 13 | "Consistency model — explain types" | 2022, 2023 Jul | 5, 5 |
| 14 | Linearizability | 2021, 2025 | 5, 4 |
| 15 | Reverse proxy — features/benefits | 2023 Dec, 2024, 2025 | 6, 5, 8 |
| 16 | Containers vs VMs — similarities/differences | 2022, 2023 Jul, 2023 Dec, 2024 | 2, 4, 4, 5 |

---

### Least Asked Topics

| Topic | Unit | Times Asked | Risk |
|-------|------|-------------|------|
| Saga Pattern | 1 | 0 | Low probability |
| Binary Translation | 2 | 0 | Low probability |
| AMD-v / Intel Virtualization | 2 | 0 | Low probability |
| IO Virtualization | 2 | 0 | Low probability |
| Jenkins Pipeline | 2 | 0 | Low probability |
| Multi-leader Replication | 3 | 0 | Low probability |
| Request Routing | 3 | 0 | Low probability |
| Raft Consensus | 4 | 0 | Low probability |
| Edge / Fog Computing | 4 | 0 | Low probability |
| Economic Denial of Sustainability (EDoS) | 4 | 1 (2023 Dec) | Emerging |
| Pub-Sub Model | 1 | 1 (2025) | Emerging |
| Cloud Migration (5Rs) | 1 | 1 (2025) | Emerging |
| Cloud Bursting | 4 | 1 (2022) | Low-Medium |
| Storage Virtualization | 3 | 1 (2021) | Low |
| Failover Architecture | 4 | 1 (2021) | Low |
| Parallel Computing | 1 | 1 (2023 Dec) | Low |

---

---

## STEP 4 — WEIGHTAGE ANALYSIS

### 1. Unit-wise Weightage (Aggregated Across All Papers)

| Unit | Total Marks Asked | % of All Marks | Avg per Paper | No. of Questions |
|------|-------------------|----------------|---------------|------------------|
| Unit 1 | ~175 | ~29% | ~29 | 22 |
| Unit 2 | ~165 | ~28% | ~28 | 22 |
| Unit 3 | ~125 | ~21% | ~21 | 17 |
| Unit 4 | ~135 | ~22% | ~23 | 28* |

*Unit 4 includes Q5 from older papers

---

### 2. Topic-wise Weightage (Top 15)

| Rank | Topic | Unit | Total Marks (All Years) |
|------|-------|------|------------------------|
| 1 | Leader Election (Ring/Modified Ring/Bully) | 4 | 53 |
| 2 | Hypervisor Types | 2 | 40 |
| 3 | Deployment Models | 1 | 38 |
| 4 | CAP Theorem | 3 | 28 |
| 5 | Zookeeper | 4 | 27 |
| 6 | Rebalancing Partitions | 3 | 26 |
| 7 | Service Models (IaaS/PaaS/SaaS) | 1 | 30 |
| 8 | IAM Security Terms | 4 | 20 |
| 9 | Replication | 3 | ~21 |
| 10 | Keystone / IAM | 4 | ~21 |
| 11 | Gluster/Lustre | 3 | 22 |
| 12 | Reverse Proxies | 4 | 19 |
| 13 | VM Migration | 2 | 19 |
| 14 | Consistency Models | 3 | 19 |
| 15 | Multitenancy / Multitenant DB | 4 | 17 |

---

### 3. Most Important Topics (Exam Priority)

| Priority | Topic | Unit | Reasons |
|----------|-------|------|---------|
| ★★★★★ | Leader Election (All 3 Algorithms) | 4 | 6/6 papers, highest marks, increasing marks each year |
| ★★★★★ | Hypervisor Types | 2 | 6/6 papers |
| ★★★★★ | CAP Theorem | 3 | 4/6 papers, consistent 5-9 marks |
| ★★★★★ | Zookeeper | 4 | 4/6 papers |
| ★★★★★ | Deployment Models | 1 | 5/6 papers |
| ★★★★ | Service Models (IaaS/PaaS/SaaS) | 1 | 5/6 papers |
| ★★★★ | Consistency Models + Linearizability | 3 | 4/6 papers |
| ★★★★ | Rebalancing Partitions | 3 | 4/6 papers |
| ★★★★ | Containers vs VMs | 2 | 5/6 papers |
| ★★★★ | Reverse Proxies | 4 | 3/6 papers (recent trend) |
| ★★★ | Gluster Architecture | 3 | 3/6 papers |
| ★★★ | VM Migration (Pre/Post copy) | 2 | 3/6 papers (recent trend) |
| ★★★ | Multitenancy / Multitenant DB | 4 | 3/6 papers |
| ★★★ | Kubernetes Architecture | 2 | 3/6 papers |
| ★★★ | Security Design Patterns | 4 | 3/6 papers |

---

### 4. Topics Repeatedly Asked for 10+ Marks

| Topic | Years with 10+ Marks | Max Marks Ever Asked |
|-------|---------------------|----------------------|
| Leader Election | 2023 Jul (10), 2024 (10), 2025 (12) | 12 |
| Hypervisor Types | 2024 (10) | 10 |
| Deployment Models | 2023 Jul (10), 2024 (10) | 10 |
| Zookeeper | 2023 Jul (10) | 10 |
| IAM Security Terms | 2021 (10), 2023 Jul (10) | 10 |
| Keystone Concepts | 2021 (10) | 10 |
| Trap & Emulate (Goldberg-Popek) | 2024 (10) | 10 |
| Gluster + Lustre | 2022 (10) | 10 |
| VM Migration | 2024 (10) | 10 |
| CAP Theorem | 2023 Jul (9) | 9 |

---

### 5. Topics Repeatedly Asked as Short Notes / Short-Answer

| Topic | Marks Range | Frequency |
|-------|-------------|-----------|
| DevOps | 4–5 | 2 |
| UnionFS / Docker | 5–6 | 2 |
| Cloud Bursting | 6 | 1 |
| Failover Architecture | 5 | 1 |
| Linearizability | 4–5 | 2 |
| Distributed Locking (problem) | 2–6 | 2 |
| MTBF/MTTR Numerical | 4–5 | 2 |
| Pub-Sub Model | 4 | 1 |
| Cloud Migration Strategies | 5 | 1 |
| Security Design Patterns | 4 | 2 |

---

---

## STEP 5 — QUESTION TREND ANALYSIS

### Directly Repeated Questions

1. **IAM Security Terms** (Cloud Time Service, Identity Mgmt, Access Mgmt, Break Glass, Key Mgmt) — **WORD-FOR-WORD identical** in 2021 and 2023 July. 10 marks each.

2. **Multitenant DB Design** (HighTechUniv / GlobalUniv) — Nearly identical in 2022 and 2023 July; concept repeated with different organizations (hospitals) in 2024.

3. **Distributed Lock Diagram** — Same diagram with same question asked in 2022 and 2023 Dec.

4. **MTBF/MTTR Numerical** — Repeated in 2021 and 2023 Dec with different numbers.

### Questions Repeated with Wording Changes

| Concept | 2021 | 2022 | 2023 Jul | 2023 Dec | 2024 | 2025 |
|---------|------|------|----------|----------|------|------|
| Leader Election | Modified Ring | Bully + Ring | Bully + Mod Ring | Ring + Mod Ring | Ring merits/demerits | All 3 compared |
| Hypervisors | Pros/Cons | Full/Para/Bare/Host | Bare Metal vs Hosted | Hosted vs Bare Metal | Types + Para/Full virt x86 | Type 1 vs Type 2 |
| CAP | — | 3 properties | State + 3 properties | 3 properties + DB choice | — | CAP + 2PC |
| Private vs Public | Why move back? | Service + Deployment models | Private/Public/Hybrid | Compare/contrast | Adv/Disadv + tools | — |
| Containers vs VMs | T/F | 1 similarity + 1 difference | 4 similarities/differences | 4 differences | Differentiate + tools | — |

### Emerging / New Topics in Recent Years

| Topic | First Appeared | Trend |
|-------|---------------|-------|
| Reverse Proxies | 2023 Dec | Asked in 2023 Dec, 2024, 2025 — STRONGLY INCREASING |
| VM Migration (Pre/Post copy) | 2023 Dec | Asked in 2023 Dec, 2024, 2025 — STRONGLY INCREASING |
| Shadow/Nested Page Tables | 2025 | New in 2025 |
| Pub-Sub / Message Queues | 2025 | New in 2025 |
| Cloud Migration Strategies (5Rs) | 2025 | New in 2025 |
| Two-Phase Commit | 2025 | New in 2025 |
| EDoS | 2023 Dec | Emerged in 2023 Dec |
| Docker Architecture | 2025 | New in 2025 |
| Goldberg-Popek / Trap & Emulate (full question) | 2024 | Appeared as major 10M question |

### Important Diagrams Repeatedly Asked

| Diagram | Times Asked | Years |
|---------|-------------|-------|
| Privilege Rings (VMM diagram with Ring 0–3) | 2 | 2022, 2023 Jul |
| Kubernetes Architecture (Master + Worker) | 2 | 2022, 2023 Dec |
| Gluster File System Architecture | 3 | 2022, 2023 Jul, 2023 Dec |
| Distributed Lock Diagram (Lease problem) | 2 | 2022, 2023 Dec |
| Ring/Modified Ring Election sketches | 2 | 2023 Dec, 2025 |
| Keystone domain/user/group/project diagram | 1 | 2025 |

### Important Comparisons Repeatedly Asked

| Comparison | Times Asked |
|------------|-------------|
| Private vs Public Cloud | 5 |
| Container vs VM | 5 |
| Bare Metal vs Hosted Hypervisor | 4 |
| Hot Migration vs Cold Migration | 3 |
| Ring vs Bully vs Modified Ring | Multiple |
| Forward Proxy vs Reverse Proxy | 2 |
| Gluster vs Lustre | 3 |
| Para Virtualization vs Full Virtualization | 2 |
| DoS vs DDoS vs EDoS | 1 |
| Elasticity vs Scalability | 3 |

---

---

## STEP 6 — PROBABILITY PREDICTION

### HIGH Probability Questions

| Probability | Topic | Expected Question | Expected Marks | Reason |
|-------------|-------|------------------|----------------|--------|
| 🔴 HIGH | Leader Election | Discuss Ring, Modified Ring and Bully algorithm. Compare all three, explain worst case messages required and merits/demerits | 10–12 | Asked in every single paper (6/6); marks increasing (5→8→10→12). Cannot be skipped. |
| 🔴 HIGH | Hypervisor Types | Explain Type 1 and Type 2 hypervisors with examples. Explain how full virtualization and para-virtualization are used for x86. | 8–10 | Asked in 6/6 papers consistently |
| 🔴 HIGH | CAP Theorem | State CAP theorem. Discuss 3 properties. Explain practical implications while choosing a database. | 6–8 | Asked in 4/6 papers, near-identical wording each time |
| 🔴 HIGH | Deployment Models | Discuss private, public, and hybrid clouds. Compare advantages and disadvantages with tools/examples. | 8–10 | Asked in 5/6 papers |
| 🔴 HIGH | Zookeeper | Explain how Zookeeper works. Common services offered. Two use cases. | 6–10 | Asked in 4/6 papers |
| 🔴 HIGH | Service Models | Explain IaaS, PaaS, SaaS with examples. Discuss advantages and disadvantages. | 8 | Asked in 5/6 papers |
| 🔴 HIGH | Rebalancing Partitions | What is rebalancing? Explain approaches (dynamic, fixed, proportional). Mention minimum requirements after repartitioning. | 5–8 | Asked in 4/6 papers |
| 🔴 HIGH | Consistency Models | What is a consistency model? Explain Linearizability, Strict, Sequential, Causal, PRAM. | 5–8 | Asked in 4/6 papers; Linearizability asked in 2021 and 2025 |
| 🔴 HIGH | Reverse Proxies | How does a reverse proxy work? Features it provides. How is it different from a forward proxy? | 6–8 | Emerged 2023 Dec, asked in 2024, 2025 — 3 consecutive years |
| 🔴 HIGH | VM Migration | Hot vs Cold migration. Explain pre-copy and post-copy. Advantages and disadvantages. | 6–10 | Asked in 2023 Dec, 2024, 2025 — clear upward trend |
| 🔴 HIGH | Containers vs VMs | Differentiate containers and VMs — at least 4 differences. Give examples of tools. | 4–6 | Asked in 5/6 papers |

---

### MEDIUM Probability Questions

| Probability | Topic | Expected Question | Expected Marks | Reason |
|-------------|-------|------------------|----------------|--------|
| 🟡 MEDIUM | Multitenancy | Define multitenancy. Benefits. Design a multitenant DB using preallocated column method. | 5–6 | Asked in 3/6 papers; exact design question repeated in 2022 & 2023 Jul |
| 🟡 MEDIUM | Kubernetes Architecture | Explain Kubernetes architecture with diagram showing master and worker components. | 6–8 | Asked in 3/6 papers with diagram |
| 🟡 MEDIUM | Gluster/Lustre | Explain Gluster file system architecture with diagram. How is it different from Lustre in metadata? | 6–8 | Asked in 3/6 papers |
| 🟡 MEDIUM | Cloud-Ready App Design | Explain 4 key design considerations to build a cloud-ready application. | 6–8 | Asked in 2/6 papers, same phrasing |
| 🟡 MEDIUM | Scalability vs Elasticity | How is scalability different from elasticity? What type of scalability exists in cloud? | 5–8 | Asked in 3/6 papers |
| 🟡 MEDIUM | Keystone/IAM | Explain domains, users, groups, projects, roles in Keystone. Token concept. | 5–8 | Asked in 4/6 papers across different sub-parts |
| 🟡 MEDIUM | Web Services & REST | Explain RESTful architecture. Explain 3 mandatory constraints. | 5–8 | Asked in 2021, 2025; likely again |
| 🟡 MEDIUM | Distributed Locking | What is the problem with distributed lock using lease? How is it overcome (fencing token)? | 4–6 | Same diagram asked in 2022, 2023 Dec |
| 🟡 MEDIUM | DevOps | What is DevOps? Benefits. How is it different from traditional dev process? | 4–5 | Asked in 2021, 2023 Jul |
| 🟡 MEDIUM | Security Design Patterns | Explain Defense in Depth and Honeypot design pattern. | 4–6 | Asked in 3/6 papers |
| 🟡 MEDIUM | Replication | Explain leader-based replication. Types of replicas. Factors influencing replica count. Also explain leaderless replication. | 5–8 | Asked in 3 forms across papers |
| 🟡 MEDIUM | IAM Security Terms | Explain: Cloud Time Service, Identity Management, Access Management, Break Glass Procedures, Key Management | 8–10 | IDENTICAL in 2021 & 2023 Jul — likely again |
| 🟡 MEDIUM | Para Virt / Privilege Rings | Explain what rings are. Explain instruction execution in the VMM diagram (Ring 0–3). | 5–8 | Asked in 2022 and 2023 Jul with same diagram |
| 🟡 MEDIUM | Fault Tolerance / MTBF/MTTR | Define fault-tolerant system. Compute MTBF and MTTR from given data. | 4–6 | Numerical asked in 2021, 2023 Dec |
| 🟡 MEDIUM | Shadow & Extended Page Tables | What are shadow page tables and extended page tables? Working of each. | 6–8 | New in 2025 — likely to appear again |

---

### LOW Probability Questions

| Probability | Topic | Expected Question | Expected Marks | Reason |
|-------------|-------|------------------|----------------|--------|
| 🟢 LOW | Parallel Computing | Explain Bit-level, Instruction-level, Task-level parallelism | 6 | Only in 2023 Dec |
| 🟢 LOW | Pub-Sub / Message Queue | Describe Message Queue model and Pub-Sub pattern | 4 | Only in 2025 |
| 🟢 LOW | Cloud Migration Strategies (5Rs) | Explain Rehost, Re-platform, Re-architect, Re-purchase, Retire, Retain | 5 | Only in 2025 |
| 🟢 LOW | UnionFS (Docker) | 3 advantages and 2 disadvantages of UnionFS | 5 | Only in 2021 |
| 🟢 LOW | Failover Architecture | Active/Active and Active/Passive failover | 5 | Only in 2021 |
| 🟢 LOW | Two-Phase Commit | Working of 2PC vs RDBMS transactions | 4 | Only in 2025 |
| 🟢 LOW | Cloud Bursting | What is cloud bursting? Benefits. | 6 | Only in 2022 |
| 🟢 LOW | DoS/DDoS/EDoS | Difference between DoS, DDoS, and EDoS | 6 | 2023 Dec only for full question |
| 🟢 LOW | Goldberg-Popek / Sensitive Instructions | Classify instructions as sensitive/control/behaviour sensitive. Design trap-and-emulate hypervisor. | 10 | Only as full question in 2024 |
| 🟢 LOW | Saga Pattern | Explain Saga pattern in microservices | — | Never asked; new topic in syllabus |
| 🟢 LOW | Raft Consensus | Explain Raft consensus algorithm | — | Never asked |
| 🟢 LOW | Edge / Fog Computing | Difference between edge and fog computing | — | Never asked |

---

---

## STEP 7 — FINAL EXAM-FOCUSED SUMMARY

---

### Most Important Topics to Study First (Priority Order)

1. **Leader Election — Ring, Modified Ring, Bully** ← Single most important topic. 6/6 papers. Must know all three, their differences, worst-case messages, and merits/demerits.
2. **Hypervisor Types** ← Second most consistent. Know Type 1 vs Type 2, full virt, para virt, x86 virtualization challenge.
3. **CAP Theorem** ← 4/6 papers, near-identical wording. Memorize 3 properties + practical implications + which DBs are CP vs AP.
4. **Deployment Models (Private/Public/Hybrid)** ← 5/6 papers. Know advantages, disadvantages, tools.
5. **Zookeeper** ← 4/6 papers. Know working, ZNode structure, services (leader election, config mgmt, naming, locking).
6. **Service Models (IaaS/PaaS/SaaS)** ← 5/6 papers. Know with examples, advantages, disadvantages.
7. **Consistency Models** ← 4/6 papers. Know Linearizability (strong consistency), Sequential, Causal, PRAM, Eventual. Know Compare-and-Set (CAS).
8. **Rebalancing Partitions** ← 4/6 papers. Know fixed, dynamic, proportional-to-nodes strategies. Know triggers for repartitioning.
9. **VM Migration** ← Strong upward trend (3 recent papers). Know hot vs cold, pre-copy vs post-copy, advantages/disadvantages.
10. **Reverse Proxies** ← 3 consecutive recent papers. Know how forward and reverse proxies work, use cases, merits/demerits.

---

### Most Probable Long Answers (10+ Marks)

| Topic | Expected Pattern | Expected Marks |
|-------|-----------------|----------------|
| Leader Election | Discuss all 3 algorithms with examples + differences + diagrams | 10–12 |
| Hypervisor Types | Types + examples + Para virt vs Full virt in x86 | 8–10 |
| Deployment Models | Private/Public/Hybrid + adv/disadv + tools | 8–10 |
| VM Migration | Hot vs Cold + Pre-copy + Post-copy + adv/disadv | 8–10 |
| Zookeeper | How it works + services + benefits + use cases | 8–10 |
| CAP Theorem + 2PC | CAP properties + implications + 2-phase commit | 8 |
| IAM Security Terms | Cloud Time Service + Identity + Access + Break Glass + Key Mgmt | 8–10 |
| Reverse Proxies | Forward proxy + Reverse proxy + working + merits/demerits | 8 |
| Gluster Architecture | Gluster diagram + comparison with Lustre | 8–10 |
| Consistency Models | Strict, Sequential, Causal, PRAM, Linearizability + CAS | 6–8 |

---

### Most Probable Short Notes (4–6 Marks)

| Topic | Expected Pattern | Expected Marks |
|-------|-----------------|----------------|
| Elasticity vs Scalability | Difference + examples | 4–6 |
| DevOps | Definition + benefits + vs traditional | 4–5 |
| Security Design Patterns | Defense in Depth + Honeypot | 4–6 |
| Distributed Locking Problem | Lease expiry issue + fencing token solution | 4–6 |
| Linearizability | Why important? + CAS operation | 4–5 |
| MTBF/MTTR Calculation | Formula + numerical | 4–5 |
| Pub-Sub Model | Definition + working | 4 |
| Cloud Migration Strategies | Rehost/Replatform/Re-architect/Retire/Retain | 5 |
| Docker UnionFS | 3 advantages + 2 disadvantages | 5 |
| Containers vs VMs | 4 differences | 4–5 |

---

### Topics with Highest Scoring Potential

| Topic | Why High Scoring Potential |
|-------|---------------------------|
| Leader Election | Very structured answers possible; 3 algorithms = clear sections; diagrams possible |
| Zookeeper | Predictable 5-point structure (ZNodes, watches, leader, services); easy to score full marks |
| CAP Theorem | Short, crisp theorem; 3 properties with real DB examples easy to remember |
| Multitenancy DB Design | Practical design question — draw table = instant marks |
| MTBF/MTTR Numerical | Formula-based; apply formula = full marks |
| Keystone IAM | Diagram given in question — just explain components |
| Gluster Architecture | Diagram = marks; memorize key components |
| Consistency Models | Memorize definitions of 4-5 models; easy partial credit |

---

### Topics Likely for Diagrams

| Topic | Diagram Description |
|-------|-------------------|
| Leader Election (Ring / Modified Ring) | Token passing in circular arrangement |
| Kubernetes Architecture | Master node + Worker node components |
| Gluster File System | GlusterFS architecture showing bricks, volumes, translators |
| Privilege Rings (VMM) | Ring 0–3 with VMM, Guest OS, Application, Physical Hardware |
| Distributed Lock (Fencing Token) | Client 1, Client 2, Lock Service, Storage interaction timeline |
| Keystone (Domain/User/Group/Project) | Hierarchical Keystone structure |
| DoS vs DDoS | Single source vs botnet attack sketch |
| Reverse Proxy vs Forward Proxy | Network diagrams |

---

### Topics Likely for Comparisons

| Comparison | Format |
|------------|--------|
| Private vs Public Cloud | Table: Security, Cost, Scalability, Control |
| Containers vs VMs | Table: Isolation, OS, Size, Boot time, Security |
| Type 1 vs Type 2 Hypervisor | Table: Position, Examples, Performance |
| Hot Migration vs Cold Migration | Table: Downtime, Complexity, When used |
| Pre-copy vs Post-copy | Table: Memory transfer, Downtime, Suitable for |
| Ring vs Bully vs Modified Ring | Table: Messages, Complexity, Fault tolerance |
| Forward Proxy vs Reverse Proxy | Table: Direction, Use cases, Who benefits |
| Gluster vs Lustre | Table: Metadata handling, Architecture, Scalability |
| Para Virt vs Full Virt | Table: Guest OS modification, Performance, Use cases |
| Elasticity vs Scalability | Table: Definition, When triggered, Cloud example |

---

### Topics Likely for Numericals

| Topic | Expected Problem Type |
|-------|----------------------|
| MTBF / MTTR | Given: downtime, total hours, failures → compute MTBF, MTTR, uptime |
| Consistent Hashing | Node placement on ring, key assignment after node join/leave |
| Ring Leader Election | Count messages in worst case for N nodes |
| Trap & Emulate (Goldberg-Popek) | Classify instructions → determine if trap-and-emulate is possible |

---

---

## FINAL PRIORITY LIST

### ★★★★★ HIGHEST EXAM IMPORTANCE — Must Study

| # | Topic | Unit | Why |
|---|-------|------|-----|
| 1 | Leader Election (Ring + Modified Ring + Bully) | 4 | 6/6 papers, increasing marks (up to 12M) |
| 2 | Hypervisor Types (Type 1, 2, Para virt, Full virt) | 2 | 6/6 papers, 5–10M range |
| 3 | CAP Theorem + Practical Implications | 3 | 4/6 papers, near-identical wording |
| 4 | Zookeeper (Working + Services) | 4 | 4/6 papers |
| 5 | Deployment Models (Private/Public/Hybrid) | 1 | 5/6 papers |
| 6 | Service Models (IaaS/PaaS/SaaS) | 1 | 5/6 papers |
| 7 | Consistency Models + Linearizability | 3 | 4/6 papers |
| 8 | Rebalancing Partitions | 3 | 4/6 papers |
| 9 | VM Migration (Hot/Cold, Pre-copy/Post-copy) | 2 | Strong recent trend 3/3 years |
| 10 | Reverse Proxies + Forward Proxies | 4 | 3/3 most recent years |
| 11 | Containers vs VMs | 2 | 5/6 papers |

---

### ★★★★ MEDIUM IMPORTANCE — Very Likely

| # | Topic | Unit |
|---|-------|------|
| 12 | Gluster File System Architecture + Lustre Comparison | 3 |
| 13 | Kubernetes Architecture Diagram | 2 |
| 14 | Multitenancy + Multitenant DB Design | 4 |
| 15 | IAM Security Terms (Cloud Time Service, Break Glass, etc.) | 4 |
| 16 | Keystone Concepts (Domain, Token, Roles, Projects) | 4 |
| 17 | Privilege Rings / VMM Diagram | 2 |
| 18 | Scalability vs Elasticity | 1 |
| 19 | Web Services & REST Principles | 1 |
| 20 | Replication (Leader-based, Leaderless) | 3 |
| 21 | Security Design Patterns (Honeypot, Defense in Depth) | 4 |
| 22 | Distributed Locking (Fencing Token) | 4 |
| 23 | Fault Tolerance + MTBF/MTTR Numerical | 4 |
| 24 | Shadow Page Tables + Extended Page Tables | 2 |
| 25 | DevOps vs Traditional | 2 |

---

### ★★★ MEDIUM IMPORTANCE — Moderately Likely

| # | Topic | Unit |
|---|-------|------|
| 26 | Cloud-Ready Application Design (4 key steps) | 1 |
| 27 | Microservices vs Monolithic | 1 |
| 28 | Docker Architecture / UnionFS | 2 |
| 29 | Goldberg-Popek (Sensitive Instructions Classification) | 2 |
| 30 | Cloud Threats: DoS / DDoS / EDoS | 4 |
| 31 | Pub-Sub / Message Queue Model | 1 |
| 32 | Cloud Migration Strategies (5Rs/6Rs) | 1 |
| 33 | Failover Architecture (Active/Active, Active/Passive) | 4 |

---

### ★★ LOW IMPORTANCE — Study Only After Above

| # | Topic | Unit |
|---|-------|------|
| 34 | Parallel Computing (Bit/Instruction/Task Level) | 1 |
| 35 | Storage Virtualization | 3 |
| 36 | Two-Phase Commit | 3 |
| 37 | Cloud Bursting | 4 |
| 38 | Saga Pattern | 1 |
| 39 | Raft Consensus | 4 |
| 40 | Edge / Fog Computing | 4 |
| 41 | Jenkins Pipeline | 2 |
| 42 | Multi-leader Replication | 3 |
| 43 | Binary Translation | 2 |

---

*Report generated from 6 PYQ papers (2021–2025) | Total 68 questions analysed | PES University Cloud Computing*
